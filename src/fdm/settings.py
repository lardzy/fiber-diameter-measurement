from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import json
import os
import re
import runpy
import sys
from uuid import uuid4

from fdm.atomic_io import atomic_write_json
from fdm.models import (
    CalibrationPreset,
    OverlayTextAnchorAlignment,
    OverlayTextSizeSpace,
)


DEFAULT_MEASUREMENT_LABEL_COLOR = "#FF0000"


class MeasurementEndpointStyle:
    CIRCLE = "circle"
    ARROW_INSIDE = "arrow_inside"
    ARROW_OUTSIDE = "arrow_outside"
    BAR = "bar"
    NONE = "none"


class OpenImageViewMode:
    DEFAULT = "default"
    FIT = "fit"
    ACTUAL = "actual"


class ScaleOverlayPlacementMode:
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"
    MANUAL = "manual"


class ScaleOverlayStyle:
    LINE = "line"
    TICKS = "ticks"
    BAR = "bar"


class AppThemeMode:
    SYSTEM = "system"
    DARK = "dark"
    LIGHT = "light"


def normalize_theme_mode(value: str | None) -> str:
    token = str(value or "").strip().lower()
    if token in {
        AppThemeMode.SYSTEM,
        AppThemeMode.DARK,
        AppThemeMode.LIGHT,
    }:
        return token
    return AppThemeMode.DARK


@dataclass(slots=True)
class MeasurementLabelStyleSettings:
    """Display-only defaults for one family of measurement result labels."""

    enabled: bool = True
    font_family: str = "Microsoft YaHei UI"
    font_size: int = 14
    color: str = DEFAULT_MEASUREMENT_LABEL_COLOR
    decimals: int = 2
    background_enabled: bool = True
    parallel_to_line: bool = False

    @staticmethod
    def _bounded_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
        try:
            numeric = int(round(float(value)))
        except (TypeError, ValueError, OverflowError):
            numeric = int(default)
        return max(int(minimum), min(int(maximum), numeric))

    def normalized_copy(self) -> "MeasurementLabelStyleSettings":
        defaults = MeasurementLabelStyleSettings()
        return MeasurementLabelStyleSettings(
            enabled=bool(self.enabled),
            font_family=str(self.font_family or defaults.font_family).strip()
            or defaults.font_family,
            font_size=self._bounded_int(
                self.font_size,
                default=defaults.font_size,
                minimum=8,
                maximum=96,
            ),
            color=str(self.color or defaults.color).strip() or defaults.color,
            decimals=self._bounded_int(
                self.decimals,
                default=defaults.decimals,
                minimum=0,
                maximum=8,
            ),
            background_enabled=bool(self.background_enabled),
            parallel_to_line=bool(self.parallel_to_line),
        )

    def to_dict(self) -> dict[str, object]:
        normalized = self.normalized_copy()
        return {
            "enabled": normalized.enabled,
            "font_family": normalized.font_family,
            "font_size": normalized.font_size,
            "color": normalized.color,
            "decimals": normalized.decimals,
            "background_enabled": normalized.background_enabled,
            "parallel_to_line": normalized.parallel_to_line,
        }

    @classmethod
    def from_dict(
        cls,
        payload: object,
        *,
        fallback: "MeasurementLabelStyleSettings | None" = None,
    ) -> "MeasurementLabelStyleSettings":
        base = (fallback or cls()).normalized_copy()
        if not isinstance(payload, dict):
            return base
        return cls(
            enabled=bool(payload.get("enabled", base.enabled)),
            font_family=str(payload.get("font_family", base.font_family)),
            font_size=cls._bounded_int(
                payload.get("font_size"),
                default=base.font_size,
                minimum=8,
                maximum=96,
            ),
            color=str(payload.get("color", base.color)),
            decimals=cls._bounded_int(
                payload.get("decimals"),
                default=base.decimals,
                minimum=0,
                maximum=8,
            ),
            background_enabled=bool(
                payload.get("background_enabled", base.background_enabled)
            ),
            parallel_to_line=bool(payload.get("parallel_to_line", base.parallel_to_line)),
        ).normalized_copy()


_MEASUREMENT_LABEL_STYLE_UNSET = object()
_MEASUREMENT_LABEL_VISIBILITY_UNSET = object()


class FocusStackProfile:
    SHARP = "sharp"
    BALANCED = "balanced"
    SOFT = "soft"


DIGITAL_SLIDE_PROFILE_FILE_KIND = "fdm.digital_slide_acquisition_profile"
DIGITAL_SLIDE_PROFILE_FILE_VERSION = 1
DIGITAL_SLIDE_PROFILE_FIELDS = (
    "digital_slide_preview_max_width",
    "digital_slide_capture_max_width",
    "digital_slide_capture_tile_codec",
    "digital_slide_capture_jpeg_quality",
    "digital_slide_xy_soft_limit",
    "digital_slide_z_soft_limit",
    "digital_slide_xy_jog_step",
    "digital_slide_z_jog_step",
    "digital_slide_z_capture_step",
    "digital_slide_jog_rate",
    "digital_slide_motor_output_enabled",
    "digital_slide_x_stage_step",
    "digital_slide_y_stage_step",
    "digital_slide_reverse_x_axis",
    "digital_slide_reverse_y_axis",
    "digital_slide_overlap_percent",
    "digital_slide_pixel_stride_mode",
    "digital_slide_x_pixel_stride",
    "digital_slide_y_pixel_stride",
    "digital_slide_blend_width",
    "digital_slide_xy_settle_ms",
    "digital_slide_xy_post_settle_ms",
    "digital_slide_z_settle_ms",
    "digital_slide_z_post_settle_ms",
    "digital_slide_first_tile_extra_wait_ms",
    "digital_slide_discard_frames",
)


@dataclass(slots=True)
class DigitalSlideAcquisitionProfile:
    """Named, portable capture parameters for one optical configuration."""

    profile_id: str
    name: str
    values: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": str(self.profile_id),
            "name": str(self.name),
            "values": {
                key: self.values[key]
                for key in DIGITAL_SLIDE_PROFILE_FIELDS
                if key in self.values
            },
        }

    @classmethod
    def from_dict(cls, payload: object) -> "DigitalSlideAcquisitionProfile":
        if not isinstance(payload, dict):
            raise ValueError("采集配置不是有效的对象。")
        raw_values = payload.get("values", {})
        if not isinstance(raw_values, dict):
            raise ValueError("采集配置参数不是有效的对象。")
        return cls(
            profile_id=str(payload.get("id", "")).strip(),
            name=str(payload.get("name", "")).strip(),
            values={
                key: raw_values[key]
                for key in DIGITAL_SLIDE_PROFILE_FIELDS
                if key in raw_values
            },
        )


class MagicSegmentModelVariant:
    EDGE_SAM = "edge_sam"
    EDGE_SAM_3X = "edge_sam_3x"


class ComplexMagicSegmentModelVariant:
    LIGHT_HQ_SAM = "light_hq_sam"
    EFFICIENTSAM_S = "efficientsam_s"


@dataclass(slots=True)
class OfflineSegmentationEnginePack:
    engine_id: str
    display_name: str
    version: str
    path: str
    manifest_sha256: str = ""
    device: str = "cpu"
    managed: bool = False

    def normalized_copy(self) -> "OfflineSegmentationEnginePack":
        engine_id = str(self.engine_id or "").strip().lower()
        if engine_id not in {"sam3", "micro_sam"}:
            raise ValueError(f"不支持的离线分割引擎：{self.engine_id}")
        path = str(self.path or "").strip()
        if not path:
            raise ValueError("离线分割引擎路径不能为空。")
        manifest_sha256 = str(self.manifest_sha256 or "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", manifest_sha256):
            raise ValueError("离线分割引擎 manifest SHA-256 无效。")
        if str(self.device or "cpu").strip().lower() != "cpu":
            raise ValueError("离线分割引擎配置必须提供纯 CPU 路径。")
        return OfflineSegmentationEnginePack(
            engine_id=engine_id,
            display_name=str(self.display_name or engine_id).strip()[:120] or engine_id,
            version=str(self.version or "unknown").strip()[:80] or "unknown",
            path=str(Path(path).expanduser()),
            manifest_sha256=manifest_sha256,
            device="cpu",
            managed=bool(self.managed),
        )

    def to_dict(self) -> dict[str, object]:
        value = self.normalized_copy()
        return {
            "engine_id": value.engine_id,
            "display_name": value.display_name,
            "version": value.version,
            "path": value.path,
            "manifest_sha256": value.manifest_sha256,
            "device": value.device,
            "managed": value.managed,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "OfflineSegmentationEnginePack":
        if not isinstance(payload, dict):
            raise ValueError("离线分割引擎配置不是有效对象。")
        return cls(
            engine_id=str(payload.get("engine_id", "")),
            display_name=str(payload.get("display_name", "")),
            version=str(payload.get("version", "")),
            path=str(payload.get("path", "")),
            manifest_sha256=str(payload.get("manifest_sha256", "")),
            device=str(payload.get("device", "cpu")),
            managed=bool(payload.get("managed", False)),
        ).normalized_copy()


class MagicSegmentToolMode:
    STANDARD = "magic_segment"
    REFERENCE = "reference_propagation"
    FIBER_QUICK = "fiber_quick"
    COMPLEX = REFERENCE


def is_magic_segment_tool_mode(value: str | None) -> bool:
    return str(value or "").strip() == MagicSegmentToolMode.STANDARD


def is_reference_propagation_tool_mode(value: str | None) -> bool:
    return str(value or "").strip() == MagicSegmentToolMode.REFERENCE


def is_fiber_quick_tool_mode(value: str | None) -> bool:
    return str(value or "").strip() == MagicSegmentToolMode.FIBER_QUICK


def is_magic_toolbar_tool_mode(value: str | None) -> bool:
    return str(value or "").strip() in {
        MagicSegmentToolMode.STANDARD,
        MagicSegmentToolMode.REFERENCE,
        MagicSegmentToolMode.FIBER_QUICK,
    }


@dataclass(slots=True)
class AreaModelMapping:
    model_name: str
    model_file: str

    def to_dict(self) -> dict[str, str]:
        return {
            "model_name": self.model_name,
            "model_file": self.model_file,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "AreaModelMapping":
        return cls(
            model_name=str(payload.get("model_name", "")).strip(),
            model_file=str(payload.get("model_file", "")).strip(),
        )


class AreaInferDevice:
    CPU = "cpu"
    AUTO = "auto"
    CUDA_0 = "cuda:0"


class RawRecordDataSource:
    DIAMETER_RESULT = "diameter_result"
    AREA_RESULT = "area_result"
    MEASUREMENT_FIELD = "measurement_field"
    UNIQUE_FIELD_RANGE = "unique_field_range"


class RawRecordMeasurementFilter:
    ALL = "all"
    LINE = "line"
    AREA = "area"
    POLYLINE = "polyline"
    COUNT = "count"


class RawRecordExportDirection:
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"


SUPPORTED_RAW_RECORD_TEMPLATE_SUFFIXES = {".xlsx", ".xlsm", ".xltx", ".xltm"}


@dataclass(slots=True)
class RawRecordExportRule:
    data_source: str = RawRecordDataSource.DIAMETER_RESULT
    field_name: str = "结果"
    measurement_filter: str = RawRecordMeasurementFilter.ALL
    sheet_name: str = "Sheet1"
    start_cell: str = "B2"
    end_cell: str = ""
    direction: str = RawRecordExportDirection.VERTICAL

    def normalized_copy(self) -> "RawRecordExportRule":
        return RawRecordExportRule(
            data_source=self._normalize_data_source(self.data_source),
            field_name=str(self.field_name or "结果").strip() or "结果",
            measurement_filter=self._normalize_measurement_filter(self.measurement_filter),
            sheet_name=str(self.sheet_name or "Sheet1").strip() or "Sheet1",
            start_cell=str(self.start_cell or "B2").strip().upper() or "B2",
            end_cell=str(self.end_cell or "").strip().upper(),
            direction=self._normalize_direction(self.direction),
        )

    def to_dict(self) -> dict[str, str]:
        normalized = self.normalized_copy()
        return {
            "data_source": normalized.data_source,
            "field_name": normalized.field_name,
            "measurement_filter": normalized.measurement_filter,
            "sheet_name": normalized.sheet_name,
            "start_cell": normalized.start_cell,
            "end_cell": normalized.end_cell,
            "direction": normalized.direction,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "RawRecordExportRule":
        return cls(
            data_source=cls._normalize_data_source(str(payload.get("data_source", RawRecordDataSource.DIAMETER_RESULT))),
            field_name=str(payload.get("field_name", "结果")).strip() or "结果",
            measurement_filter=cls._normalize_measurement_filter(str(payload.get("measurement_filter", RawRecordMeasurementFilter.ALL))),
            sheet_name=str(payload.get("sheet_name", "Sheet1")).strip() or "Sheet1",
            start_cell=str(payload.get("start_cell", "B2")).strip().upper() or "B2",
            end_cell=str(payload.get("end_cell", "")).strip().upper(),
            direction=cls._normalize_direction(str(payload.get("direction", RawRecordExportDirection.VERTICAL))),
        ).normalized_copy()

    @staticmethod
    def _normalize_data_source(value: str | None) -> str:
        token = str(value or "").strip()
        if token in {
            RawRecordDataSource.DIAMETER_RESULT,
            RawRecordDataSource.AREA_RESULT,
            RawRecordDataSource.MEASUREMENT_FIELD,
            RawRecordDataSource.UNIQUE_FIELD_RANGE,
        }:
            return token
        return RawRecordDataSource.DIAMETER_RESULT

    @staticmethod
    def _normalize_measurement_filter(value: str | None) -> str:
        token = str(value or "").strip()
        if token in {
            RawRecordMeasurementFilter.ALL,
            RawRecordMeasurementFilter.LINE,
            RawRecordMeasurementFilter.AREA,
            RawRecordMeasurementFilter.POLYLINE,
            RawRecordMeasurementFilter.COUNT,
        }:
            return token
        return RawRecordMeasurementFilter.ALL

    @staticmethod
    def _normalize_direction(value: str | None) -> str:
        token = str(value or "").strip()
        if token in {
            RawRecordExportDirection.VERTICAL,
            RawRecordExportDirection.HORIZONTAL,
        }:
            return token
        return RawRecordExportDirection.VERTICAL


@dataclass(slots=True)
class RawRecordTemplate:
    name: str
    path: str
    rules: list[RawRecordExportRule] = field(default_factory=lambda: [RawRecordExportRule()])

    def normalized_copy(self) -> "RawRecordTemplate":
        path_token = normalize_raw_record_template_path(self.path)
        name = str(self.name or "").strip()
        if not name and path_token:
            name = Path(path_token).stem
        return RawRecordTemplate(
            name=name,
            path=path_token,
            rules=[rule.normalized_copy() for rule in self.rules if isinstance(rule, RawRecordExportRule)],
        )

    def to_dict(self) -> dict[str, object]:
        normalized = self.normalized_copy()
        return {
            "name": normalized.name,
            "path": normalized.path,
            "rules": [rule.to_dict() for rule in normalized.rules],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "RawRecordTemplate":
        rules_payload = payload.get("rules", [])
        rules = [
            RawRecordExportRule.from_dict(item)
            for item in rules_payload
            if isinstance(item, dict)
        ] if isinstance(rules_payload, list) else []
        return cls(
            name=str(payload.get("name", "")).strip(),
            path=str(payload.get("path", "")).strip(),
            rules=rules or [RawRecordExportRule()],
        ).normalized_copy()


def is_supported_raw_record_template_path(value: str | Path | None) -> bool:
    token = str(value or "").strip()
    return Path(token).suffix.lower() in SUPPORTED_RAW_RECORD_TEMPLATE_SUFFIXES


def project_runtime_root() -> Path:
    return Path(__file__).resolve().parents[2]


def application_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return project_runtime_root()


def bundle_resource_root() -> Path:
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", "")
        if meipass:
            return Path(str(meipass)).resolve()
        internal = application_root() / "_internal"
        if internal.exists():
            return internal.resolve()
    return project_runtime_root()


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _display_path(path: Path) -> str:
    return path.as_posix() if not path.is_absolute() else str(path)


def _to_relative_path(value: str | Path | None, *, root: Path) -> str:
    token = str(value or "").strip()
    if not token:
        return ""
    path = Path(token).expanduser()
    if not path.is_absolute():
        return _display_path(path)
    if _path_is_within(path, root):
        return path.resolve().relative_to(root.resolve()).as_posix()
    return str(path.resolve())


def to_app_relative_path(value: str | Path | None) -> str:
    return _to_relative_path(value, root=application_root())


def to_resource_relative_path(value: str | Path | None) -> str:
    return _to_relative_path(value, root=bundle_resource_root())


def normalize_raw_record_template_path(value: str | Path | None) -> str:
    token = to_resource_relative_path(value)
    if not token or not is_supported_raw_record_template_path(token):
        return ""
    return token


def _resolve_relative_path(value: str | Path | None, *, root: Path, default: str | Path | None = None) -> Path:
    token = str(value or "").strip()
    if not token and default is not None:
        token = str(default).strip()
    if not token:
        return Path()
    path = Path(token).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def resolve_app_relative_path(value: str | Path | None, *, default: str | Path | None = None) -> Path:
    return _resolve_relative_path(value, root=application_root(), default=default)


def resolve_resource_relative_path(value: str | Path | None, *, default: str | Path | None = None) -> Path:
    return _resolve_relative_path(value, root=bundle_resource_root(), default=default)


def runtime_directory() -> Path:
    return bundle_resource_root() / "runtime"


def default_area_reference_root() -> Path:
    runtime_candidate = runtime_directory() / "area-infer"
    if runtime_candidate.exists():
        return runtime_candidate
    return project_runtime_root() / ".tmp" / "textile-device-monitor-ref" / "textile-device-monitor" / "area-infer"


def default_area_vendor_root() -> str:
    candidate = default_area_reference_root() / "vendor" / "yolact"
    return to_resource_relative_path(candidate) if candidate.exists() else ""


def default_area_weights_directory() -> str:
    project_candidate = runtime_directory() / "area-models"
    if project_candidate.exists():
        return to_resource_relative_path(project_candidate)
    project_candidate = project_runtime_root() / ".tmp" / "area-models"
    if project_candidate.exists():
        return to_resource_relative_path(project_candidate)
    return str(settings_directory() / "area-models")


def default_area_worker_python() -> str:
    if getattr(sys, "frozen", False):
        executable = Path(sys.executable).resolve()
        worker_exe = executable.with_name("FiberAreaWorker.exe")
        if worker_exe.exists():
            return to_app_relative_path(worker_exe)
        return "FiberAreaWorker.exe"
    return ""


def default_area_model_mappings() -> list[AreaModelMapping]:
    metadata_path = default_area_reference_root() / "app" / "model_metadata.py"
    if metadata_path.is_file():
        try:
            namespace = runpy.run_path(str(metadata_path))
            specs = namespace.get("MODEL_SPECS")
            if isinstance(specs, (list, tuple)):
                mappings = [
                    AreaModelMapping(
                        model_name=str(item.get("model_name") or "").strip(),
                        model_file=str(item.get("model_file") or "").strip(),
                    )
                    for item in specs
                    if isinstance(item, dict)
                    and str(item.get("model_name") or "").strip()
                    and str(item.get("model_file") or "").strip()
                ]
                if mappings:
                    return mappings
        except (OSError, RuntimeError, TypeError, ValueError):
            pass
    return [
        AreaModelMapping(model_name="粘纤-莱赛尔", model_file="b_v1_1.3.pth"),
        AreaModelMapping(model_name="棉-粘纤", model_file="b_cv_1.3.pth"),
        AreaModelMapping(model_name="棉-莱赛尔", model_file="b_c1_1.3.pth"),
        AreaModelMapping(model_name="棉-莫代尔", model_file="b_cm_1.3.pth"),
        AreaModelMapping(model_name="棉-再生纤维素纤维", model_file="b_cc_1.3.pth"),
        AreaModelMapping(model_name="棉-粘-莱-莫", model_file="b_cvlm_1.3.pth"),
    ]


@dataclass(slots=True)
class WorkspaceLayoutSettings:
    """User-owned workbench dimensions and collapsible-panel preferences."""

    version: int = 3
    project_width: int = 260
    inspector_width: int = 380
    results_height: int = 260
    inspector_records_height: int = 260
    statistics_expanded: bool = False
    calibration_expanded: bool = True
    records_expanded: bool = True
    area_recognition_expanded: bool = False
    object_properties_expanded: bool = False

    @staticmethod
    def _extent(value: object, default: int) -> int:
        try:
            numeric = int(round(float(value)))
        except (TypeError, ValueError, OverflowError):
            numeric = int(default)
        return max(120, min(2000, numeric))

    def normalized_copy(self) -> "WorkspaceLayoutSettings":
        return WorkspaceLayoutSettings(
            version=3,
            project_width=self._extent(self.project_width, 260),
            inspector_width=max(376, self._extent(self.inspector_width, 380)),
            results_height=self._extent(self.results_height, 260),
            inspector_records_height=self._extent(self.inspector_records_height, 260),
            statistics_expanded=bool(self.statistics_expanded),
            calibration_expanded=bool(self.calibration_expanded),
            records_expanded=bool(self.records_expanded),
            area_recognition_expanded=bool(self.area_recognition_expanded),
            object_properties_expanded=bool(self.object_properties_expanded),
        )

    def to_dict(self) -> dict[str, object]:
        normalized = self.normalized_copy()
        return {
            "version": normalized.version,
            "project_width": normalized.project_width,
            "inspector_width": normalized.inspector_width,
            "results_height": normalized.results_height,
            "inspector_records_height": normalized.inspector_records_height,
            "statistics_expanded": normalized.statistics_expanded,
            "calibration_expanded": normalized.calibration_expanded,
            "records_expanded": normalized.records_expanded,
            "area_recognition_expanded": normalized.area_recognition_expanded,
            "object_properties_expanded": normalized.object_properties_expanded,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "WorkspaceLayoutSettings":
        if not isinstance(payload, dict):
            return cls()
        defaults = cls()
        return cls(
            version=3,
            project_width=cls._extent(payload.get("project_width"), defaults.project_width),
            inspector_width=max(
                376,
                cls._extent(payload.get("inspector_width"), defaults.inspector_width),
            ),
            results_height=cls._extent(payload.get("results_height"), defaults.results_height),
            inspector_records_height=cls._extent(
                payload.get("inspector_records_height"),
                defaults.inspector_records_height,
            ),
            statistics_expanded=bool(payload.get("statistics_expanded", defaults.statistics_expanded)),
            calibration_expanded=bool(
                payload.get("calibration_expanded", defaults.calibration_expanded)
            ),
            records_expanded=bool(payload.get("records_expanded", defaults.records_expanded)),
            area_recognition_expanded=bool(
                payload.get("area_recognition_expanded", defaults.area_recognition_expanded)
            ),
            object_properties_expanded=bool(
                payload.get("object_properties_expanded", defaults.object_properties_expanded)
            ),
        )


@dataclass(slots=True)
class AppSettings:
    theme_mode: str = AppThemeMode.DARK
    length_measurement_label_style: MeasurementLabelStyleSettings = field(
        default=_MEASUREMENT_LABEL_STYLE_UNSET  # type: ignore[arg-type]
    )
    area_measurement_label_style: MeasurementLabelStyleSettings = field(
        default=_MEASUREMENT_LABEL_STYLE_UNSET  # type: ignore[arg-type]
    )
    # Deprecated flat aliases are retained for callers constructing AppSettings
    # directly. The typed styles are canonical once construction completes;
    # persistence and rendering never read later mutations of these aliases.
    show_measurement_labels: bool = field(
        default=_MEASUREMENT_LABEL_VISIBILITY_UNSET  # type: ignore[arg-type]
    )
    measurement_label_font_family: str = "Microsoft YaHei UI"
    measurement_label_font_size: int = 14
    measurement_label_color: str = DEFAULT_MEASUREMENT_LABEL_COLOR
    measurement_label_decimals: int = 2
    measurement_label_parallel_to_line: bool = False
    measurement_label_background_enabled: bool = True
    show_count_numbers: bool = False
    count_number_font_family: str = "Microsoft YaHei UI"
    count_number_font_size: int = 12
    count_number_color: str = "#FFFFFF"
    measurement_endpoint_style: str = MeasurementEndpointStyle.BAR
    default_measurement_color: str = "#2A9D8F"
    open_image_view_mode: str = OpenImageViewMode.FIT
    scale_overlay_placement_mode: str = ScaleOverlayPlacementMode.BOTTOM_RIGHT
    scale_overlay_style: str = ScaleOverlayStyle.TICKS
    scale_overlay_length_value: float = 50.0
    scale_overlay_color: str = "#F4F1DE"
    scale_overlay_text_color: str = "#F4F1DE"
    scale_overlay_font_family: str = "Microsoft YaHei UI"
    scale_overlay_font_size: int = 18
    text_font_family: str = "Microsoft YaHei UI"
    text_font_size: int = 18
    text_color: str = "#F7F4EA"
    text_size_space: str = OverlayTextSizeSpace.IMAGE_PX
    text_anchor_alignment: str = OverlayTextAnchorAlignment.CENTER
    overlay_line_color: str = "#F7F4EA"
    overlay_line_width: float = 2.5
    show_canvas_navigator: bool = True
    object_snap_enabled: bool = True
    object_snap_kinds: list[str] = field(
        default_factory=lambda: [
            "point",
            "endpoint",
            "midpoint",
            "center",
            "quadrant",
            "intersection",
        ]
    )
    object_snap_aperture_px: float = 10.0
    focus_stack_profile: str = FocusStackProfile.BALANCED
    focus_stack_sharpen_strength: int = 35
    magic_segment_model_variant: str = MagicSegmentModelVariant.EDGE_SAM_3X
    magic_segment_fill_draft_holes_enabled: bool = False
    magic_segment_standard_roi_enabled: bool = False
    magic_segment_standard_add_roi_enabled: bool = False
    magic_segment_standard_subtract_roi_enabled: bool = True
    magic_segment_standard_subtract_input_mode: str = "smart"
    magic_segment_restrict_subtract_roi_to_primary_bounds: bool = True
    magic_segment_small_object_subtract_enhancement_enabled: bool = True
    magic_segment_small_object_roi_area_threshold_px: int = 160000
    fiber_quick_roi_enabled: bool = True
    fiber_quick_edge_trim_enabled: bool = True
    fiber_quick_line_extension_px: float = 0.0
    offline_segmentation_engine_packs: list[OfflineSegmentationEnginePack] = field(
        default_factory=list
    )
    main_window_geometry: str = ""
    main_window_state: str = ""
    measurement_results_header_state: str = ""
    inspector_measurement_results_header_state: str = ""
    workspace_layout: WorkspaceLayoutSettings = field(default_factory=WorkspaceLayoutSettings)
    main_window_is_maximized: bool = False
    recent_export_dir: str = ""
    recent_project_dir: str = ""
    area_model_mappings: list[AreaModelMapping] = field(default_factory=default_area_model_mappings)
    area_weights_dir: str = field(default_factory=default_area_weights_directory)
    area_vendor_root: str = field(default_factory=default_area_vendor_root)
    area_worker_python: str = field(default_factory=default_area_worker_python)
    area_infer_device: str = AreaInferDevice.CPU
    calibration_presets: list[CalibrationPreset] = field(default_factory=list)
    load_issues: list[dict[str, object]] = field(default_factory=list, repr=False, compare=False)
    selected_capture_device_id: str = ""
    raw_record_templates: list[RawRecordTemplate] = field(default_factory=list)
    last_raw_record_template_path: str = ""
    digital_slide_last_output_path: str = ""
    digital_slide_preview_max_width: int = 1280
    digital_slide_capture_max_width: int = 1600
    digital_slide_capture_tile_codec: str = "png"
    digital_slide_capture_jpeg_quality: int = 90
    digital_slide_xy_soft_limit: int = 1_000_000
    digital_slide_z_soft_limit: int = 200_000
    digital_slide_xy_jog_step: int = 5000
    digital_slide_z_jog_step: int = 1000
    digital_slide_z_capture_lower: int | None = None
    digital_slide_z_capture_upper: int | None = None
    digital_slide_z_capture_step: int = 1000
    digital_slide_jog_rate: int = 12
    digital_slide_motor_output_enabled: bool = True
    digital_slide_x_stage_step: int = 5000
    digital_slide_y_stage_step: int = 5000
    digital_slide_reverse_x_axis: bool = False
    digital_slide_reverse_y_axis: bool = False
    digital_slide_overlap_percent: int = 0
    digital_slide_pixel_stride_mode: str = "auto_overlap"
    digital_slide_x_pixel_stride: int = 1280
    digital_slide_y_pixel_stride: int = 960
    digital_slide_blend_width: int = 0
    digital_slide_xy_settle_ms: int = 200
    digital_slide_xy_post_settle_ms: int = 100
    digital_slide_z_settle_ms: int = 80
    digital_slide_z_post_settle_ms: int = 40
    digital_slide_first_tile_extra_wait_ms: int = 3000
    digital_slide_discard_frames: int = 2
    digital_slide_focus_wheel_step: int = 1
    digital_slide_dynamic_focus_overview_enabled: bool = True
    digital_slide_smooth_navigation_enabled: bool = True
    digital_slide_shift_navigation_enabled: bool = False
    digital_slide_profiles: list[DigitalSlideAcquisitionProfile] = field(default_factory=list)
    digital_slide_active_profile_id: str = ""

    def __post_init__(self) -> None:
        legacy_visibility_was_explicit = (
            self.show_measurement_labels is not _MEASUREMENT_LABEL_VISIBILITY_UNSET
        )
        legacy_style = MeasurementLabelStyleSettings(
            enabled=(
                bool(self.show_measurement_labels)
                if legacy_visibility_was_explicit
                else True
            ),
            font_family=self.measurement_label_font_family,
            font_size=self.measurement_label_font_size,
            color=self.measurement_label_color,
            decimals=self.measurement_label_decimals,
            background_enabled=self.measurement_label_background_enabled,
            parallel_to_line=self.measurement_label_parallel_to_line,
        ).normalized_copy()
        if self.length_measurement_label_style is _MEASUREMENT_LABEL_STYLE_UNSET:
            self.length_measurement_label_style = replace(legacy_style)
        else:
            self.length_measurement_label_style = (
                self.length_measurement_label_style.normalized_copy()
            )
        if self.area_measurement_label_style is _MEASUREMENT_LABEL_STYLE_UNSET:
            # Keep fresh installations visually quiet while preserving the
            # deprecated direct-constructor alias when callers explicitly use
            # it. Existing persisted settings are restored by ``from_dict``.
            self.area_measurement_label_style = replace(
                legacy_style,
                enabled=(
                    legacy_style.enabled
                    if legacy_visibility_was_explicit
                    else False
                ),
            )
        else:
            self.area_measurement_label_style = self.area_measurement_label_style.normalized_copy()
        self._sync_legacy_measurement_label_fields()

    def _sync_legacy_measurement_label_fields(self) -> None:
        style = self.length_measurement_label_style.normalized_copy()
        self.show_measurement_labels = style.enabled
        self.measurement_label_font_family = style.font_family
        self.measurement_label_font_size = style.font_size
        self.measurement_label_color = style.color
        self.measurement_label_decimals = style.decimals
        self.measurement_label_parallel_to_line = style.parallel_to_line
        self.measurement_label_background_enabled = style.background_enabled

    def normalized_copy(self) -> "AppSettings":
        normalized = replace(self)
        normalized.workspace_layout = self.workspace_layout.normalized_copy()
        normalized.theme_mode = normalize_theme_mode(self.theme_mode)
        normalized.length_measurement_label_style = (
            self.length_measurement_label_style.normalized_copy()
        )
        normalized.area_measurement_label_style = (
            self.area_measurement_label_style.normalized_copy()
        )
        normalized._sync_legacy_measurement_label_fields()
        normalized.count_number_font_size = self._normalize_font_size(self.count_number_font_size, minimum=8, maximum=96)
        normalized.measurement_endpoint_style = self._normalize_measurement_endpoint_style(self.measurement_endpoint_style)
        normalized.open_image_view_mode = self._normalize_open_image_view_mode(self.open_image_view_mode)
        normalized.scale_overlay_placement_mode = self._normalize_scale_overlay_placement_mode(self.scale_overlay_placement_mode)
        normalized.scale_overlay_style = self._normalize_scale_overlay_style(self.scale_overlay_style)
        normalized.scale_overlay_length_value = self._normalize_scale_overlay_length_value(self.scale_overlay_length_value)
        normalized.scale_overlay_font_size = self._normalize_font_size(self.scale_overlay_font_size, minimum=8, maximum=96)
        normalized.text_font_size = self._normalize_font_size(self.text_font_size, minimum=8, maximum=144)
        normalized.text_size_space = OverlayTextSizeSpace.normalize(
            self.text_size_space
        )
        normalized.text_anchor_alignment = OverlayTextAnchorAlignment.normalize(
            self.text_anchor_alignment
        )
        normalized.overlay_line_width = self._normalize_overlay_line_width(self.overlay_line_width)
        normalized.show_canvas_navigator = bool(self.show_canvas_navigator)
        normalized.object_snap_enabled = bool(self.object_snap_enabled)
        allowed_snap_kinds = {
            "point",
            "endpoint",
            "midpoint",
            "center",
            "quadrant",
            "intersection",
            "nearest",
        }
        normalized.object_snap_kinds = list(
            dict.fromkeys(
                kind
                for kind in (str(item).strip().lower() for item in self.object_snap_kinds)
                if kind in allowed_snap_kinds
            )
        )
        try:
            snap_aperture = float(self.object_snap_aperture_px)
        except (TypeError, ValueError):
            snap_aperture = 10.0
        normalized.object_snap_aperture_px = max(4.0, min(40.0, snap_aperture))
        normalized.focus_stack_profile = self._normalize_focus_stack_profile(self.focus_stack_profile)
        normalized.focus_stack_sharpen_strength = self._normalize_focus_stack_sharpen_strength(self.focus_stack_sharpen_strength)
        normalized.magic_segment_model_variant = self._normalize_magic_segment_model_variant(self.magic_segment_model_variant)
        normalized.magic_segment_standard_subtract_input_mode = self._normalize_magic_segment_standard_subtract_input_mode(
            self.magic_segment_standard_subtract_input_mode
        )
        normalized.magic_segment_small_object_roi_area_threshold_px = self._normalize_magic_small_object_roi_area_threshold_px(
            self.magic_segment_small_object_roi_area_threshold_px
        )
        normalized.fiber_quick_line_extension_px = self._normalize_fiber_quick_line_extension_px(self.fiber_quick_line_extension_px)
        normalized.offline_segmentation_engine_packs = []
        seen_engine_ids: set[str] = set()
        for pack in self.offline_segmentation_engine_packs:
            try:
                normalized_pack = (
                    pack.normalized_copy()
                    if isinstance(pack, OfflineSegmentationEnginePack)
                    else OfflineSegmentationEnginePack.from_dict(pack)
                )
            except (TypeError, ValueError):
                continue
            if normalized_pack.engine_id in seen_engine_ids:
                continue
            seen_engine_ids.add(normalized_pack.engine_id)
            normalized.offline_segmentation_engine_packs.append(normalized_pack)
        normalized.recent_export_dir = self._normalize_recent_directory(self.recent_export_dir)
        normalized.recent_project_dir = self._normalize_recent_directory(self.recent_project_dir)
        normalized.area_weights_dir = self._normalize_weights_dir(self.area_weights_dir)
        normalized.area_vendor_root = self._normalize_vendor_root(self.area_vendor_root)
        normalized.area_worker_python = self._normalize_worker_program(self.area_worker_python)
        normalized.area_infer_device = self._normalize_area_infer_device(self.area_infer_device)
        normalized.raw_record_templates = self._normalize_raw_record_templates(self.raw_record_templates)
        normalized.last_raw_record_template_path = normalize_raw_record_template_path(self.last_raw_record_template_path)
        normalized.digital_slide_last_output_path = self._normalize_digital_slide_output_path(self.digital_slide_last_output_path)
        normalized.digital_slide_preview_max_width = self._normalize_optional_width(self.digital_slide_preview_max_width, default=1280)
        normalized.digital_slide_capture_max_width = self._normalize_optional_width(self.digital_slide_capture_max_width, default=1600)
        normalized.digital_slide_capture_tile_codec = self._normalize_digital_slide_tile_codec(self.digital_slide_capture_tile_codec)
        normalized.digital_slide_capture_jpeg_quality = self._normalize_int_range(
            self.digital_slide_capture_jpeg_quality,
            default=90,
            minimum=70,
            maximum=95,
        )
        normalized.digital_slide_xy_soft_limit = self._normalize_int_range(self.digital_slide_xy_soft_limit, default=1_000_000, minimum=0, maximum=10_000_000)
        normalized.digital_slide_z_soft_limit = self._normalize_int_range(self.digital_slide_z_soft_limit, default=200_000, minimum=0, maximum=10_000_000)
        normalized.digital_slide_xy_jog_step = self._normalize_int_range(self.digital_slide_xy_jog_step, default=5000, minimum=1, maximum=1_000_000)
        normalized.digital_slide_z_jog_step = self._normalize_int_range(self.digital_slide_z_jog_step, default=1000, minimum=1, maximum=1_000_000)
        normalized.digital_slide_z_capture_lower = self._normalize_optional_signed_int(self.digital_slide_z_capture_lower, minimum=-10_000_000, maximum=10_000_000)
        normalized.digital_slide_z_capture_upper = self._normalize_optional_signed_int(self.digital_slide_z_capture_upper, minimum=-10_000_000, maximum=10_000_000)
        normalized.digital_slide_z_capture_step = self._normalize_int_range(self.digital_slide_z_capture_step, default=1000, minimum=1, maximum=1_000_000)
        normalized.digital_slide_jog_rate = self._normalize_int_range(self.digital_slide_jog_rate, default=12, minimum=1, maximum=50)
        normalized.digital_slide_x_stage_step = self._normalize_signed_int_range(self.digital_slide_x_stage_step, default=5000, minimum=-10_000_000, maximum=10_000_000)
        normalized.digital_slide_y_stage_step = self._normalize_signed_int_range(self.digital_slide_y_stage_step, default=5000, minimum=-10_000_000, maximum=10_000_000)
        normalized.digital_slide_reverse_x_axis = bool(self.digital_slide_reverse_x_axis)
        normalized.digital_slide_reverse_y_axis = bool(self.digital_slide_reverse_y_axis)
        normalized.digital_slide_overlap_percent = self._normalize_int_range(self.digital_slide_overlap_percent, default=0, minimum=0, maximum=90)
        normalized.digital_slide_pixel_stride_mode = self._normalize_digital_slide_pixel_stride_mode(self.digital_slide_pixel_stride_mode)
        normalized.digital_slide_x_pixel_stride = self._normalize_int_range(self.digital_slide_x_pixel_stride, default=1280, minimum=1, maximum=100_000)
        normalized.digital_slide_y_pixel_stride = self._normalize_int_range(self.digital_slide_y_pixel_stride, default=960, minimum=1, maximum=100_000)
        normalized.digital_slide_blend_width = self._normalize_int_range(self.digital_slide_blend_width, default=0, minimum=0, maximum=10_000)
        normalized.digital_slide_xy_settle_ms = self._normalize_int_range(self.digital_slide_xy_settle_ms, default=200, minimum=0, maximum=10_000)
        normalized.digital_slide_xy_post_settle_ms = self._normalize_int_range(self.digital_slide_xy_post_settle_ms, default=100, minimum=0, maximum=5000)
        normalized.digital_slide_z_settle_ms = self._normalize_int_range(self.digital_slide_z_settle_ms, default=80, minimum=0, maximum=10_000)
        normalized.digital_slide_z_post_settle_ms = self._normalize_int_range(self.digital_slide_z_post_settle_ms, default=40, minimum=0, maximum=5000)
        normalized.digital_slide_first_tile_extra_wait_ms = self._normalize_int_range(
            self.digital_slide_first_tile_extra_wait_ms,
            default=3000,
            minimum=0,
            maximum=60_000,
        )
        normalized.digital_slide_discard_frames = self._normalize_int_range(self.digital_slide_discard_frames, default=2, minimum=0, maximum=20)
        normalized.digital_slide_focus_wheel_step = self._normalize_int_range(self.digital_slide_focus_wheel_step, default=1, minimum=1, maximum=10)
        normalized.digital_slide_dynamic_focus_overview_enabled = bool(
            self.digital_slide_dynamic_focus_overview_enabled
        )
        normalized.digital_slide_smooth_navigation_enabled = bool(
            self.digital_slide_smooth_navigation_enabled
        )
        normalized.digital_slide_shift_navigation_enabled = bool(
            self.digital_slide_shift_navigation_enabled
        )
        profiles = self._normalize_digital_slide_profiles(
            self.digital_slide_profiles,
            fallback=normalized,
        )
        requested_active = str(self.digital_slide_active_profile_id or "").strip()
        profile_ids = {profile.profile_id for profile in profiles}
        active_id = requested_active if requested_active in profile_ids else ""
        if not profiles:
            profiles = [
                DigitalSlideAcquisitionProfile(
                    profile_id="default",
                    name="默认配置",
                    values=self._digital_slide_profile_values_from_settings(normalized),
                )
            ]
            active_id = "default"
        elif not active_id:
            active_id = profiles[0].profile_id
            # When a profile collection has no valid active id, its first
            # entry is authoritative rather than unrelated legacy flat values.
            self._apply_digital_slide_profile_values(normalized, profiles[0].values)
        active_values = self._digital_slide_profile_values_from_settings(normalized)
        normalized.digital_slide_profiles = [
            DigitalSlideAcquisitionProfile(
                profile_id=profile.profile_id,
                name=profile.name,
                values=(active_values if profile.profile_id == active_id else dict(profile.values)),
            )
            for profile in profiles
        ]
        normalized.digital_slide_active_profile_id = active_id
        return normalized

    @staticmethod
    def _normalize_digital_slide_profile_name(value: object) -> str:
        token = " ".join(str(value or "").strip().split())
        token = "".join(character for character in token if ord(character) >= 32)
        return token[:80]

    @classmethod
    def _normalize_digital_slide_profile_values(
        cls,
        values: object,
        *,
        fallback: "AppSettings",
    ) -> dict[str, object]:
        source = values if isinstance(values, dict) else {}

        def item(name: str) -> object:
            return source.get(name, getattr(fallback, name))

        return {
            "digital_slide_preview_max_width": cls._normalize_optional_width(item("digital_slide_preview_max_width"), default=1280),
            "digital_slide_capture_max_width": cls._normalize_optional_width(item("digital_slide_capture_max_width"), default=1600),
            "digital_slide_capture_tile_codec": cls._normalize_digital_slide_tile_codec(item("digital_slide_capture_tile_codec")),
            "digital_slide_capture_jpeg_quality": cls._normalize_int_range(item("digital_slide_capture_jpeg_quality"), default=90, minimum=70, maximum=95),
            "digital_slide_xy_soft_limit": cls._normalize_int_range(item("digital_slide_xy_soft_limit"), default=1_000_000, minimum=0, maximum=10_000_000),
            "digital_slide_z_soft_limit": cls._normalize_int_range(item("digital_slide_z_soft_limit"), default=200_000, minimum=0, maximum=10_000_000),
            "digital_slide_xy_jog_step": cls._normalize_int_range(item("digital_slide_xy_jog_step"), default=5000, minimum=1, maximum=1_000_000),
            "digital_slide_z_jog_step": cls._normalize_int_range(item("digital_slide_z_jog_step"), default=1000, minimum=1, maximum=1_000_000),
            "digital_slide_z_capture_step": cls._normalize_int_range(item("digital_slide_z_capture_step"), default=1000, minimum=1, maximum=1_000_000),
            "digital_slide_jog_rate": cls._normalize_int_range(item("digital_slide_jog_rate"), default=12, minimum=1, maximum=50),
            "digital_slide_motor_output_enabled": bool(item("digital_slide_motor_output_enabled")),
            "digital_slide_x_stage_step": cls._normalize_signed_int_range(item("digital_slide_x_stage_step"), default=5000, minimum=-10_000_000, maximum=10_000_000),
            "digital_slide_y_stage_step": cls._normalize_signed_int_range(item("digital_slide_y_stage_step"), default=5000, minimum=-10_000_000, maximum=10_000_000),
            "digital_slide_reverse_x_axis": bool(item("digital_slide_reverse_x_axis")),
            "digital_slide_reverse_y_axis": bool(item("digital_slide_reverse_y_axis")),
            "digital_slide_overlap_percent": cls._normalize_int_range(item("digital_slide_overlap_percent"), default=0, minimum=0, maximum=90),
            "digital_slide_pixel_stride_mode": cls._normalize_digital_slide_pixel_stride_mode(item("digital_slide_pixel_stride_mode")),
            "digital_slide_x_pixel_stride": cls._normalize_int_range(item("digital_slide_x_pixel_stride"), default=1280, minimum=1, maximum=100_000),
            "digital_slide_y_pixel_stride": cls._normalize_int_range(item("digital_slide_y_pixel_stride"), default=960, minimum=1, maximum=100_000),
            "digital_slide_blend_width": cls._normalize_int_range(item("digital_slide_blend_width"), default=0, minimum=0, maximum=10_000),
            "digital_slide_xy_settle_ms": cls._normalize_int_range(item("digital_slide_xy_settle_ms"), default=200, minimum=0, maximum=10_000),
            "digital_slide_xy_post_settle_ms": cls._normalize_int_range(item("digital_slide_xy_post_settle_ms"), default=100, minimum=0, maximum=5000),
            "digital_slide_z_settle_ms": cls._normalize_int_range(item("digital_slide_z_settle_ms"), default=80, minimum=0, maximum=10_000),
            "digital_slide_z_post_settle_ms": cls._normalize_int_range(item("digital_slide_z_post_settle_ms"), default=40, minimum=0, maximum=5000),
            "digital_slide_first_tile_extra_wait_ms": cls._normalize_int_range(item("digital_slide_first_tile_extra_wait_ms"), default=3000, minimum=0, maximum=60_000),
            "digital_slide_discard_frames": cls._normalize_int_range(item("digital_slide_discard_frames"), default=2, minimum=0, maximum=20),
        }

    @classmethod
    def _normalize_digital_slide_profiles(
        cls,
        profiles: object,
        *,
        fallback: "AppSettings",
    ) -> list[DigitalSlideAcquisitionProfile]:
        if not isinstance(profiles, (list, tuple)):
            return []
        normalized: list[DigitalSlideAcquisitionProfile] = []
        used_ids: set[str] = set()
        used_names: set[str] = set()
        for raw_profile in profiles:
            try:
                profile = (
                    raw_profile
                    if isinstance(raw_profile, DigitalSlideAcquisitionProfile)
                    else DigitalSlideAcquisitionProfile.from_dict(raw_profile)
                )
            except (TypeError, ValueError):
                continue
            name = cls._normalize_digital_slide_profile_name(profile.name)
            if not name:
                continue
            base_name = name
            suffix = 2
            while name.casefold() in used_names:
                name = f"{base_name} ({suffix})"
                suffix += 1
            profile_id = str(profile.profile_id or "").strip()
            if not profile_id or profile_id in used_ids:
                profile_id = uuid4().hex
            used_ids.add(profile_id)
            used_names.add(name.casefold())
            normalized.append(
                DigitalSlideAcquisitionProfile(
                    profile_id=profile_id,
                    name=name,
                    values=cls._normalize_digital_slide_profile_values(
                        profile.values,
                        fallback=fallback,
                    ),
                )
            )
        return normalized

    @staticmethod
    def _digital_slide_profile_values_from_settings(settings: "AppSettings") -> dict[str, object]:
        return {name: getattr(settings, name) for name in DIGITAL_SLIDE_PROFILE_FIELDS}

    @staticmethod
    def _apply_digital_slide_profile_values(
        settings: "AppSettings",
        values: dict[str, object],
    ) -> None:
        for name in DIGITAL_SLIDE_PROFILE_FIELDS:
            if name in values:
                setattr(settings, name, values[name])

    def activate_digital_slide_profile(self, profile_id: str) -> bool:
        normalized = self.normalized_copy()
        target = next(
            (
                profile
                for profile in normalized.digital_slide_profiles
                if profile.profile_id == str(profile_id)
            ),
            None,
        )
        if target is None:
            return False
        self.digital_slide_profiles = [
            DigitalSlideAcquisitionProfile(
                profile_id=profile.profile_id,
                name=profile.name,
                values=dict(profile.values),
            )
            for profile in normalized.digital_slide_profiles
        ]
        self.digital_slide_active_profile_id = target.profile_id
        self._apply_digital_slide_profile_values(self, target.values)
        return True

    def active_digital_slide_profile(self) -> DigitalSlideAcquisitionProfile:
        normalized = self.normalized_copy()
        return next(
            profile
            for profile in normalized.digital_slide_profiles
            if profile.profile_id == normalized.digital_slide_active_profile_id
        )

    def resolved_area_weights_dir(self) -> Path:
        return resolve_resource_relative_path(
            self.area_weights_dir,
            default=default_area_weights_directory(),
        )

    def resolved_area_vendor_root(self) -> Path:
        return resolve_resource_relative_path(
            self.area_vendor_root,
            default=default_area_vendor_root(),
        )

    def resolved_area_worker_program(self) -> str:
        token = str(self._normalize_worker_program(self.area_worker_python)).strip()
        if not token:
            return ""
        return str(resolve_app_relative_path(token))

    @staticmethod
    def _normalize_weights_dir(value: str | Path | None) -> str:
        default_token = default_area_weights_directory()
        token = to_resource_relative_path(value)
        resolved = resolve_resource_relative_path(token, default=default_token)
        if getattr(sys, "frozen", False):
            legacy_default = legacy_area_weights_directory().resolve()
            default_resolved = resolve_resource_relative_path(default_token, default=default_token)
            if token and Path(str(token)).expanduser().is_absolute():
                absolute_value = Path(str(token)).expanduser().resolve()
                if absolute_value == legacy_default and default_resolved.exists():
                    return default_token
        if not resolved.exists():
            return default_token
        return token or default_token

    @staticmethod
    def _normalize_vendor_root(value: str | Path | None) -> str:
        token = to_resource_relative_path(value)
        resolved = resolve_resource_relative_path(token, default=default_area_vendor_root())
        if not resolved.exists():
            return default_area_vendor_root()
        return token or default_area_vendor_root()

    @staticmethod
    def _normalize_worker_program(value: str | Path | None) -> str:
        token = str(value or "").strip()
        default_token = default_area_worker_python()
        if not token:
            return default_token
        path = Path(token).expanduser()
        if path.is_absolute():
            resolved = path.resolve()
            if resolved == Path(sys.executable).resolve():
                return default_token
            if not resolved.exists():
                return default_token
            if default_token:
                default_resolved = resolve_app_relative_path(default_token)
                if default_resolved and default_resolved.exists() and resolved == default_resolved:
                    return default_token
            return to_app_relative_path(resolved)
        relative_token = _display_path(path)
        if default_token and not resolve_app_relative_path(relative_token).exists():
            return default_token
        return relative_token

    @staticmethod
    def _normalize_area_infer_device(value: str | None) -> str:
        token = str(value or "").strip().lower()
        if token == "cuda":
            return AreaInferDevice.CUDA_0
        if token in {AreaInferDevice.CPU, AreaInferDevice.AUTO, AreaInferDevice.CUDA_0}:
            return token
        return AreaInferDevice.CPU

    @staticmethod
    def _normalize_scale_overlay_style(value: str | None) -> str:
        token = str(value or "").strip()
        if token in {
            ScaleOverlayStyle.LINE,
            ScaleOverlayStyle.TICKS,
            ScaleOverlayStyle.BAR,
        }:
            return token
        return ScaleOverlayStyle.TICKS

    @staticmethod
    def _normalize_measurement_label_decimals(value: int | float | str | None) -> int:
        try:
            numeric = int(round(float(value)))
        except (TypeError, ValueError):
            numeric = 2
        return max(0, min(8, numeric))

    @staticmethod
    def _normalize_measurement_endpoint_style(value: str | None) -> str:
        token = str(value or "").strip()
        if token in {
            MeasurementEndpointStyle.CIRCLE,
            MeasurementEndpointStyle.ARROW_INSIDE,
            MeasurementEndpointStyle.ARROW_OUTSIDE,
            MeasurementEndpointStyle.BAR,
            MeasurementEndpointStyle.NONE,
        }:
            return token
        return MeasurementEndpointStyle.BAR

    @staticmethod
    def _normalize_open_image_view_mode(value: str | None) -> str:
        token = str(value or "").strip()
        if token in {
            OpenImageViewMode.DEFAULT,
            OpenImageViewMode.FIT,
            OpenImageViewMode.ACTUAL,
        }:
            return token
        return OpenImageViewMode.FIT

    @staticmethod
    def _normalize_scale_overlay_placement_mode(value: str | None) -> str:
        token = str(value or "").strip()
        if token in {
            ScaleOverlayPlacementMode.TOP_LEFT,
            ScaleOverlayPlacementMode.TOP_RIGHT,
            ScaleOverlayPlacementMode.BOTTOM_LEFT,
            ScaleOverlayPlacementMode.BOTTOM_RIGHT,
            ScaleOverlayPlacementMode.MANUAL,
        }:
            return token
        return ScaleOverlayPlacementMode.BOTTOM_RIGHT

    @staticmethod
    def _normalize_scale_overlay_length_value(value: float | int | str | None) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 50.0
        return max(0.01, min(1_000_000.0, numeric))

    @staticmethod
    def _normalize_font_size(value: int | float | str | None, *, minimum: int, maximum: int) -> int:
        try:
            numeric = int(round(float(value)))
        except (TypeError, ValueError):
            numeric = minimum
        return max(minimum, min(maximum, numeric))

    @staticmethod
    def _normalize_focus_stack_profile(value: str | None) -> str:
        token = str(value or "").strip()
        if token in {
            FocusStackProfile.SHARP,
            FocusStackProfile.BALANCED,
            FocusStackProfile.SOFT,
        }:
            return token
        return FocusStackProfile.BALANCED

    @staticmethod
    def _normalize_focus_stack_sharpen_strength(value: int | float | str | None) -> int:
        try:
            numeric = int(round(float(value)))
        except (TypeError, ValueError):
            numeric = 35
        return max(0, min(100, numeric))

    @staticmethod
    def _normalize_magic_segment_model_variant(value: str | None) -> str:
        token = str(value or "").strip()
        if token in {
            MagicSegmentModelVariant.EDGE_SAM,
            MagicSegmentModelVariant.EDGE_SAM_3X,
        }:
            return token
        return MagicSegmentModelVariant.EDGE_SAM_3X

    @staticmethod
    def _normalize_magic_small_object_roi_area_threshold_px(value: int | float | str | None) -> int:
        try:
            numeric = int(round(float(value)))
        except (TypeError, ValueError):
            numeric = 160000
        return max(4096, min(4_000_000, numeric))

    @staticmethod
    def _normalize_magic_segment_standard_subtract_input_mode(value: object) -> str:
        mode = str(value or "").strip().lower()
        if mode in {"smart", "polygon", "freehand"}:
            return mode
        return "smart"

    @staticmethod
    def _normalize_fiber_quick_line_extension_px(value: int | float | str | None) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 0.0
        return max(-20.0, min(20.0, numeric))

    @staticmethod
    def _normalize_overlay_line_width(value: int | float | str | None) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 2.5
        return max(0.5, min(24.0, numeric))

    @staticmethod
    def _normalize_recent_directory(value: str | Path | None) -> str:
        token = str(value or "").strip()
        if not token:
            return ""
        try:
            path = Path(token).expanduser()
            resolved = path.resolve() if path.exists() else path
        except (OSError, RuntimeError):
            return token
        if resolved.exists() and resolved.is_file():
            return str(resolved.parent)
        return str(resolved)

    @staticmethod
    def _normalize_raw_record_templates(value: list[RawRecordTemplate] | None) -> list[RawRecordTemplate]:
        normalized_templates: list[RawRecordTemplate] = []
        seen_paths: set[str] = set()
        for template in value or []:
            normalized = template.normalized_copy()
            if not normalized.path:
                continue
            key = normalized.path.casefold()
            if key in seen_paths:
                continue
            seen_paths.add(key)
            normalized_templates.append(normalized)
        return normalized_templates

    @staticmethod
    def _normalize_int_range(value: int | float | str | None, *, default: int, minimum: int, maximum: int) -> int:
        try:
            numeric = int(round(float(value)))
        except (TypeError, ValueError):
            numeric = int(default)
        return max(int(minimum), min(int(maximum), numeric))

    @staticmethod
    def _normalize_signed_int_range(value: int | float | str | None, *, default: int, minimum: int, maximum: int) -> int:
        try:
            numeric = int(round(float(value)))
        except (TypeError, ValueError):
            numeric = int(default)
        return max(int(minimum), min(int(maximum), numeric))

    @staticmethod
    def _normalize_optional_signed_int(value: int | float | str | None, *, minimum: int, maximum: int) -> int | None:
        if value is None:
            return None
        token = str(value).strip()
        if not token:
            return None
        try:
            numeric = int(round(float(token)))
        except (TypeError, ValueError):
            return None
        return max(int(minimum), min(int(maximum), numeric))

    @staticmethod
    def _normalize_optional_width(value: int | float | str | None, *, default: int) -> int:
        try:
            numeric = int(round(float(value)))
        except (TypeError, ValueError):
            numeric = int(default)
        if numeric <= 0:
            return 0
        return max(320, min(20_000, numeric))

    @staticmethod
    def _normalize_digital_slide_output_path(value: str | Path | None) -> str:
        token = str(value or "").strip()
        if not token:
            return ""
        try:
            return str(Path(token).expanduser())
        except (OSError, RuntimeError):
            return token

    @staticmethod
    def _normalize_digital_slide_pixel_stride_mode(value: str | None) -> str:
        token = str(value or "").strip()
        if token in {"auto_overlap", "manual_pixels"}:
            return token
        return "auto_overlap"

    @staticmethod
    def _normalize_digital_slide_tile_codec(value: str | None) -> str:
        token = str(value or "").strip().lower()
        if token in {"jpg", "jpeg"}:
            return "jpeg"
        return "png"

    def to_dict(self) -> dict[str, object]:
        normalized = self.normalized_copy()
        return {
            "version": 3,
            "theme_mode": normalized.theme_mode,
            "length_measurement_label_style": normalized.length_measurement_label_style.to_dict(),
            "area_measurement_label_style": normalized.area_measurement_label_style.to_dict(),
            # Keep flat aliases for older application builds. They mirror the
            # length style and are ignored when typed style payloads exist.
            "show_measurement_labels": normalized.show_measurement_labels,
            "measurement_label_font_family": normalized.measurement_label_font_family,
            "measurement_label_font_size": normalized.measurement_label_font_size,
            "measurement_label_color": normalized.measurement_label_color,
            "measurement_label_decimals": normalized.measurement_label_decimals,
            "measurement_label_parallel_to_line": normalized.measurement_label_parallel_to_line,
            "measurement_label_background_enabled": normalized.measurement_label_background_enabled,
            "show_count_numbers": normalized.show_count_numbers,
            "count_number_font_family": normalized.count_number_font_family,
            "count_number_font_size": normalized.count_number_font_size,
            "count_number_color": normalized.count_number_color,
            "measurement_endpoint_style": normalized.measurement_endpoint_style,
            "default_measurement_color": normalized.default_measurement_color,
            "open_image_view_mode": normalized.open_image_view_mode,
            "scale_overlay_placement_mode": normalized.scale_overlay_placement_mode,
            "scale_overlay_style": normalized.scale_overlay_style,
            "scale_overlay_length_value": normalized.scale_overlay_length_value,
            "scale_overlay_color": normalized.scale_overlay_color,
            "scale_overlay_text_color": normalized.scale_overlay_text_color,
            "scale_overlay_font_family": normalized.scale_overlay_font_family,
            "scale_overlay_font_size": normalized.scale_overlay_font_size,
            "text_font_family": normalized.text_font_family,
            "text_font_size": normalized.text_font_size,
            "text_color": normalized.text_color,
            "text_size_space": normalized.text_size_space,
            "text_anchor_alignment": normalized.text_anchor_alignment,
            "overlay_line_color": normalized.overlay_line_color,
            "overlay_line_width": normalized.overlay_line_width,
            "show_canvas_navigator": normalized.show_canvas_navigator,
            "object_snap_enabled": normalized.object_snap_enabled,
            "object_snap_kinds": list(normalized.object_snap_kinds),
            "object_snap_aperture_px": normalized.object_snap_aperture_px,
            "focus_stack_profile": normalized.focus_stack_profile,
            "focus_stack_sharpen_strength": normalized.focus_stack_sharpen_strength,
            "magic_segment_model_variant": normalized.magic_segment_model_variant,
            "magic_segment_fill_draft_holes_enabled": normalized.magic_segment_fill_draft_holes_enabled,
            "magic_segment_standard_roi_enabled": normalized.magic_segment_standard_add_roi_enabled,
            "magic_segment_standard_add_roi_enabled": normalized.magic_segment_standard_add_roi_enabled,
            "magic_segment_standard_subtract_roi_enabled": normalized.magic_segment_standard_subtract_roi_enabled,
            "magic_segment_standard_subtract_input_mode": normalized.magic_segment_standard_subtract_input_mode,
            "magic_segment_restrict_subtract_roi_to_primary_bounds": normalized.magic_segment_restrict_subtract_roi_to_primary_bounds,
            "magic_segment_small_object_subtract_enhancement_enabled": normalized.magic_segment_small_object_subtract_enhancement_enabled,
            "magic_segment_small_object_roi_area_threshold_px": normalized.magic_segment_small_object_roi_area_threshold_px,
            "fiber_quick_roi_enabled": normalized.fiber_quick_roi_enabled,
            "fiber_quick_edge_trim_enabled": normalized.fiber_quick_edge_trim_enabled,
            "fiber_quick_line_extension_px": normalized.fiber_quick_line_extension_px,
            "offline_segmentation_engine_packs": [
                pack.to_dict()
                for pack in normalized.offline_segmentation_engine_packs
            ],
            "main_window_geometry": normalized.main_window_geometry,
            "main_window_state": normalized.main_window_state,
            "measurement_results_header_state": normalized.measurement_results_header_state,
            "inspector_measurement_results_header_state": normalized.inspector_measurement_results_header_state,
            "workspace_layout": normalized.workspace_layout.to_dict(),
            "main_window_is_maximized": normalized.main_window_is_maximized,
            "recent_export_dir": normalized.recent_export_dir,
            "recent_project_dir": normalized.recent_project_dir,
            "area_model_mappings": [item.to_dict() for item in normalized.area_model_mappings],
            "area_weights_dir": normalized.area_weights_dir,
            "area_vendor_root": normalized.area_vendor_root,
            "area_worker_python": normalized.area_worker_python,
            "area_infer_device": normalized.area_infer_device,
            "calibration_presets": [preset.to_dict() for preset in normalized.calibration_presets],
            "selected_capture_device_id": normalized.selected_capture_device_id,
            "raw_record_templates": [template.to_dict() for template in normalized.raw_record_templates],
            "last_raw_record_template_path": normalized.last_raw_record_template_path,
            "digital_slide_last_output_path": normalized.digital_slide_last_output_path,
            "digital_slide_preview_max_width": normalized.digital_slide_preview_max_width,
            "digital_slide_capture_max_width": normalized.digital_slide_capture_max_width,
            "digital_slide_capture_tile_codec": normalized.digital_slide_capture_tile_codec,
            "digital_slide_capture_jpeg_quality": normalized.digital_slide_capture_jpeg_quality,
            "digital_slide_xy_soft_limit": normalized.digital_slide_xy_soft_limit,
            "digital_slide_z_soft_limit": normalized.digital_slide_z_soft_limit,
            "digital_slide_xy_jog_step": normalized.digital_slide_xy_jog_step,
            "digital_slide_z_jog_step": normalized.digital_slide_z_jog_step,
            "digital_slide_z_capture_lower": normalized.digital_slide_z_capture_lower,
            "digital_slide_z_capture_upper": normalized.digital_slide_z_capture_upper,
            "digital_slide_z_capture_step": normalized.digital_slide_z_capture_step,
            "digital_slide_jog_rate": normalized.digital_slide_jog_rate,
            "digital_slide_motor_output_enabled": normalized.digital_slide_motor_output_enabled,
            "digital_slide_x_stage_step": normalized.digital_slide_x_stage_step,
            "digital_slide_y_stage_step": normalized.digital_slide_y_stage_step,
            "digital_slide_reverse_x_axis": normalized.digital_slide_reverse_x_axis,
            "digital_slide_reverse_y_axis": normalized.digital_slide_reverse_y_axis,
            "digital_slide_overlap_percent": normalized.digital_slide_overlap_percent,
            "digital_slide_pixel_stride_mode": normalized.digital_slide_pixel_stride_mode,
            "digital_slide_x_pixel_stride": normalized.digital_slide_x_pixel_stride,
            "digital_slide_y_pixel_stride": normalized.digital_slide_y_pixel_stride,
            "digital_slide_blend_width": normalized.digital_slide_blend_width,
            "digital_slide_xy_settle_ms": normalized.digital_slide_xy_settle_ms,
            "digital_slide_xy_post_settle_ms": normalized.digital_slide_xy_post_settle_ms,
            "digital_slide_z_settle_ms": normalized.digital_slide_z_settle_ms,
            "digital_slide_z_post_settle_ms": normalized.digital_slide_z_post_settle_ms,
            "digital_slide_first_tile_extra_wait_ms": normalized.digital_slide_first_tile_extra_wait_ms,
            "digital_slide_discard_frames": normalized.digital_slide_discard_frames,
            "digital_slide_focus_wheel_step": normalized.digital_slide_focus_wheel_step,
            "digital_slide_dynamic_focus_overview_enabled": normalized.digital_slide_dynamic_focus_overview_enabled,
            "digital_slide_smooth_navigation_enabled": normalized.digital_slide_smooth_navigation_enabled,
            "digital_slide_shift_navigation_enabled": normalized.digital_slide_shift_navigation_enabled,
            "digital_slide_profiles": [
                profile.to_dict()
                for profile in normalized.digital_slide_profiles
            ],
            "digital_slide_active_profile_id": normalized.digital_slide_active_profile_id,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "AppSettings":
        settings = cls()
        settings.theme_mode = normalize_theme_mode(payload.get("theme_mode", settings.theme_mode))
        legacy_style = MeasurementLabelStyleSettings(
            enabled=bool(payload.get("show_measurement_labels", settings.show_measurement_labels)),
            font_family=str(
                payload.get(
                    "measurement_label_font_family",
                    settings.measurement_label_font_family,
                )
            ),
            font_size=cls._normalize_font_size(
                payload.get(
                    "measurement_label_font_size",
                    settings.measurement_label_font_size,
                ),
                minimum=8,
                maximum=96,
            ),
            color=str(
                payload.get("measurement_label_color", settings.measurement_label_color)
            ),
            decimals=cls._normalize_measurement_label_decimals(
                payload.get(
                    "measurement_label_decimals",
                    settings.measurement_label_decimals,
                )
            ),
            parallel_to_line=bool(
                payload.get(
                    "measurement_label_parallel_to_line",
                    settings.measurement_label_parallel_to_line,
                )
            ),
            background_enabled=bool(
                payload.get(
                    "measurement_label_background_enabled",
                    settings.measurement_label_background_enabled,
                )
            ),
        ).normalized_copy()
        settings.length_measurement_label_style = MeasurementLabelStyleSettings.from_dict(
            payload.get("length_measurement_label_style"),
            fallback=legacy_style,
        )
        area_fallback = replace(
            legacy_style,
            enabled=(
                legacy_style.enabled
                if "show_measurement_labels" in payload
                else settings.area_measurement_label_style.enabled
            ),
        )
        settings.area_measurement_label_style = MeasurementLabelStyleSettings.from_dict(
            payload.get("area_measurement_label_style"),
            fallback=area_fallback,
        )
        settings._sync_legacy_measurement_label_fields()
        settings.show_count_numbers = bool(payload.get("show_count_numbers", settings.show_count_numbers))
        settings.count_number_font_family = str(payload.get("count_number_font_family", settings.count_number_font_family))
        settings.count_number_font_size = cls._normalize_font_size(
            payload.get("count_number_font_size", settings.count_number_font_size),
            minimum=8,
            maximum=96,
        )
        settings.count_number_color = str(payload.get("count_number_color", settings.count_number_color))
        settings.measurement_endpoint_style = cls._normalize_measurement_endpoint_style(
            payload.get("measurement_endpoint_style", settings.measurement_endpoint_style)
        )
        settings.default_measurement_color = str(payload.get("default_measurement_color", settings.default_measurement_color))
        settings.open_image_view_mode = cls._normalize_open_image_view_mode(
            payload.get("open_image_view_mode", settings.open_image_view_mode)
        )
        settings.scale_overlay_placement_mode = cls._normalize_scale_overlay_placement_mode(
            payload.get("scale_overlay_placement_mode", settings.scale_overlay_placement_mode)
        )
        settings.scale_overlay_style = cls._normalize_scale_overlay_style(payload.get("scale_overlay_style", settings.scale_overlay_style))
        settings.scale_overlay_length_value = cls._normalize_scale_overlay_length_value(
            payload.get("scale_overlay_length_value", settings.scale_overlay_length_value)
        )
        settings.scale_overlay_color = str(payload.get("scale_overlay_color", settings.scale_overlay_color))
        settings.scale_overlay_text_color = str(payload.get("scale_overlay_text_color", settings.scale_overlay_text_color))
        settings.scale_overlay_font_family = str(payload.get("scale_overlay_font_family", settings.scale_overlay_font_family))
        settings.scale_overlay_font_size = cls._normalize_font_size(
            payload.get("scale_overlay_font_size", settings.scale_overlay_font_size),
            minimum=8,
            maximum=96,
        )
        settings.text_font_family = str(payload.get("text_font_family", settings.text_font_family))
        settings.text_font_size = cls._normalize_font_size(
            payload.get("text_font_size", settings.text_font_size),
            minimum=8,
            maximum=144,
        )
        settings.text_color = str(payload.get("text_color", settings.text_color))
        settings.text_size_space = OverlayTextSizeSpace.normalize(
            payload.get("text_size_space", settings.text_size_space)
        )
        settings.text_anchor_alignment = OverlayTextAnchorAlignment.normalize(
            payload.get(
                "text_anchor_alignment",
                settings.text_anchor_alignment,
            )
        )
        settings.overlay_line_color = str(payload.get("overlay_line_color", settings.overlay_line_color))
        settings.overlay_line_width = cls._normalize_overlay_line_width(
            payload.get("overlay_line_width", settings.overlay_line_width)
        )
        settings.show_canvas_navigator = bool(
            payload.get("show_canvas_navigator", settings.show_canvas_navigator)
        )
        settings.object_snap_enabled = bool(
            payload.get("object_snap_enabled", settings.object_snap_enabled)
        )
        raw_snap_kinds = payload.get("object_snap_kinds", settings.object_snap_kinds)
        if isinstance(raw_snap_kinds, (list, tuple, set)):
            settings.object_snap_kinds = [str(item) for item in raw_snap_kinds]
        try:
            settings.object_snap_aperture_px = float(
                payload.get(
                    "object_snap_aperture_px",
                    settings.object_snap_aperture_px,
                )
            )
        except (TypeError, ValueError):
            settings.object_snap_aperture_px = 10.0
        settings.focus_stack_profile = cls._normalize_focus_stack_profile(payload.get("focus_stack_profile", settings.focus_stack_profile))
        settings.focus_stack_sharpen_strength = cls._normalize_focus_stack_sharpen_strength(
            payload.get("focus_stack_sharpen_strength", settings.focus_stack_sharpen_strength)
        )
        settings.magic_segment_model_variant = cls._normalize_magic_segment_model_variant(
            payload.get("magic_segment_model_variant", settings.magic_segment_model_variant)
        )
        raw_engine_packs = payload.get("offline_segmentation_engine_packs", [])
        if isinstance(raw_engine_packs, list):
            for index, item in enumerate(raw_engine_packs):
                try:
                    settings.offline_segmentation_engine_packs.append(
                        OfflineSegmentationEnginePack.from_dict(item)
                    )
                except (TypeError, ValueError) as exc:
                    settings.load_issues.append(
                        {
                            "kind": "offline_segmentation_engine_pack",
                            "index": index,
                            "message": str(exc),
                        }
                    )
        settings.magic_segment_fill_draft_holes_enabled = bool(
            payload.get(
                "magic_segment_fill_draft_holes_enabled",
                settings.magic_segment_fill_draft_holes_enabled,
            )
        )
        legacy_standard_roi = payload.get("magic_segment_standard_roi_enabled", None)
        legacy_standard_roi_enabled = bool(legacy_standard_roi) if legacy_standard_roi is not None else None
        settings.magic_segment_standard_add_roi_enabled = bool(
            payload.get(
                "magic_segment_standard_add_roi_enabled",
                legacy_standard_roi_enabled
                if legacy_standard_roi_enabled is not None
                else settings.magic_segment_standard_add_roi_enabled,
            )
        )
        settings.magic_segment_standard_subtract_roi_enabled = bool(
            payload.get(
                "magic_segment_standard_subtract_roi_enabled",
                True if legacy_standard_roi_enabled is None else (True if legacy_standard_roi_enabled else settings.magic_segment_standard_subtract_roi_enabled),
            )
        )
        settings.magic_segment_standard_roi_enabled = settings.magic_segment_standard_add_roi_enabled
        settings.magic_segment_standard_subtract_input_mode = cls._normalize_magic_segment_standard_subtract_input_mode(
            payload.get(
                "magic_segment_standard_subtract_input_mode",
                settings.magic_segment_standard_subtract_input_mode,
            )
        )
        settings.magic_segment_restrict_subtract_roi_to_primary_bounds = bool(
            payload.get(
                "magic_segment_restrict_subtract_roi_to_primary_bounds",
                settings.magic_segment_restrict_subtract_roi_to_primary_bounds,
            )
        )
        settings.magic_segment_small_object_subtract_enhancement_enabled = bool(
            payload.get(
                "magic_segment_small_object_subtract_enhancement_enabled",
                settings.magic_segment_small_object_subtract_enhancement_enabled,
            )
        )
        settings.magic_segment_small_object_roi_area_threshold_px = cls._normalize_magic_small_object_roi_area_threshold_px(
            payload.get(
                "magic_segment_small_object_roi_area_threshold_px",
                settings.magic_segment_small_object_roi_area_threshold_px,
            )
        )
        settings.fiber_quick_roi_enabled = bool(
            payload.get(
                "fiber_quick_roi_enabled",
                settings.fiber_quick_roi_enabled,
            )
        )
        settings.fiber_quick_edge_trim_enabled = bool(
            payload.get(
                "fiber_quick_edge_trim_enabled",
                settings.fiber_quick_edge_trim_enabled,
            )
        )
        settings.fiber_quick_line_extension_px = cls._normalize_fiber_quick_line_extension_px(
            payload.get(
                "fiber_quick_line_extension_px",
                settings.fiber_quick_line_extension_px,
            )
        )
        settings.main_window_geometry = str(payload.get("main_window_geometry", settings.main_window_geometry)).strip()
        settings.main_window_state = str(payload.get("main_window_state", settings.main_window_state)).strip()
        settings.measurement_results_header_state = str(
            payload.get("measurement_results_header_state", settings.measurement_results_header_state)
        ).strip()
        settings.inspector_measurement_results_header_state = str(
            payload.get(
                "inspector_measurement_results_header_state",
                settings.inspector_measurement_results_header_state,
            )
        ).strip()
        settings.workspace_layout = WorkspaceLayoutSettings.from_dict(payload.get("workspace_layout"))
        settings.main_window_is_maximized = bool(payload.get("main_window_is_maximized", settings.main_window_is_maximized))
        settings.recent_export_dir = cls._normalize_recent_directory(payload.get("recent_export_dir", settings.recent_export_dir))
        settings.recent_project_dir = cls._normalize_recent_directory(payload.get("recent_project_dir", settings.recent_project_dir))
        mappings = payload.get("area_model_mappings", None)
        if isinstance(mappings, list):
            settings.area_model_mappings = [
                AreaModelMapping.from_dict(item)
                for item in mappings
                if isinstance(item, dict)
                and (str(item.get("model_name", "")).strip() or str(item.get("model_file", "")).strip())
            ]
        elif mappings is None:
            settings.area_model_mappings = default_area_model_mappings()
        settings.area_weights_dir = cls._normalize_weights_dir(payload.get("area_weights_dir", settings.area_weights_dir))
        settings.area_vendor_root = cls._normalize_vendor_root(payload.get("area_vendor_root", settings.area_vendor_root))
        settings.area_worker_python = cls._normalize_worker_program(payload.get("area_worker_python", settings.area_worker_python))
        settings.area_infer_device = cls._normalize_area_infer_device(
            payload.get("area_infer_device", settings.area_infer_device)
        )
        presets = payload.get("calibration_presets", None)
        if isinstance(presets, list):
            valid_presets: list[CalibrationPreset] = []
            for index, item in enumerate(presets):
                if not isinstance(item, dict) or not str(item.get("name", "")).strip():
                    continue
                try:
                    valid_presets.append(CalibrationPreset.from_dict(item))
                except (KeyError, TypeError, ValueError) as exc:
                    settings.load_issues.append(
                        {
                            "kind": "calibration_preset",
                            "index": index,
                            "message": str(exc),
                            "raw_payload": dict(item),
                        }
                    )
            settings.calibration_presets = valid_presets
        settings.selected_capture_device_id = str(payload.get("selected_capture_device_id", settings.selected_capture_device_id)).strip()
        templates = payload.get("raw_record_templates", None)
        if isinstance(templates, list):
            settings.raw_record_templates = cls._normalize_raw_record_templates(
                [
                    RawRecordTemplate.from_dict(item)
                    for item in templates
                    if isinstance(item, dict)
                ]
            )
        settings.last_raw_record_template_path = normalize_raw_record_template_path(
            payload.get("last_raw_record_template_path", settings.last_raw_record_template_path)
        )
        settings.digital_slide_last_output_path = cls._normalize_digital_slide_output_path(
            payload.get("digital_slide_last_output_path", settings.digital_slide_last_output_path)
        )
        settings.digital_slide_preview_max_width = cls._normalize_optional_width(
            payload.get("digital_slide_preview_max_width", settings.digital_slide_preview_max_width),
            default=1280,
        )
        settings.digital_slide_capture_max_width = cls._normalize_optional_width(
            payload.get("digital_slide_capture_max_width", settings.digital_slide_capture_max_width),
            default=1600,
        )
        settings.digital_slide_capture_tile_codec = cls._normalize_digital_slide_tile_codec(
            payload.get("digital_slide_capture_tile_codec", settings.digital_slide_capture_tile_codec)
        )
        settings.digital_slide_capture_jpeg_quality = cls._normalize_int_range(
            payload.get("digital_slide_capture_jpeg_quality", settings.digital_slide_capture_jpeg_quality),
            default=90,
            minimum=70,
            maximum=95,
        )
        settings.digital_slide_xy_soft_limit = cls._normalize_int_range(
            payload.get("digital_slide_xy_soft_limit", settings.digital_slide_xy_soft_limit),
            default=1_000_000,
            minimum=0,
            maximum=10_000_000,
        )
        settings.digital_slide_z_soft_limit = cls._normalize_int_range(
            payload.get("digital_slide_z_soft_limit", settings.digital_slide_z_soft_limit),
            default=200_000,
            minimum=0,
            maximum=10_000_000,
        )
        settings.digital_slide_xy_jog_step = cls._normalize_int_range(
            payload.get("digital_slide_xy_jog_step", settings.digital_slide_xy_jog_step),
            default=5000,
            minimum=1,
            maximum=1_000_000,
        )
        settings.digital_slide_z_jog_step = cls._normalize_int_range(
            payload.get("digital_slide_z_jog_step", settings.digital_slide_z_jog_step),
            default=1000,
            minimum=1,
            maximum=1_000_000,
        )
        settings.digital_slide_z_capture_lower = cls._normalize_optional_signed_int(
            payload.get("digital_slide_z_capture_lower", settings.digital_slide_z_capture_lower),
            minimum=-10_000_000,
            maximum=10_000_000,
        )
        settings.digital_slide_z_capture_upper = cls._normalize_optional_signed_int(
            payload.get("digital_slide_z_capture_upper", settings.digital_slide_z_capture_upper),
            minimum=-10_000_000,
            maximum=10_000_000,
        )
        settings.digital_slide_z_capture_step = cls._normalize_int_range(
            payload.get("digital_slide_z_capture_step", settings.digital_slide_z_capture_step),
            default=1000,
            minimum=1,
            maximum=1_000_000,
        )
        settings.digital_slide_jog_rate = cls._normalize_int_range(
            payload.get("digital_slide_jog_rate", settings.digital_slide_jog_rate),
            default=12,
            minimum=1,
            maximum=50,
        )
        settings.digital_slide_motor_output_enabled = bool(
            payload.get("digital_slide_motor_output_enabled", settings.digital_slide_motor_output_enabled)
        )
        settings.digital_slide_x_stage_step = cls._normalize_signed_int_range(
            payload.get("digital_slide_x_stage_step", settings.digital_slide_x_stage_step),
            default=5000,
            minimum=-10_000_000,
            maximum=10_000_000,
        )
        settings.digital_slide_y_stage_step = cls._normalize_signed_int_range(
            payload.get("digital_slide_y_stage_step", settings.digital_slide_y_stage_step),
            default=5000,
            minimum=-10_000_000,
            maximum=10_000_000,
        )
        settings.digital_slide_reverse_x_axis = bool(
            payload.get("digital_slide_reverse_x_axis", settings.digital_slide_reverse_x_axis)
        )
        settings.digital_slide_reverse_y_axis = bool(
            payload.get("digital_slide_reverse_y_axis", settings.digital_slide_reverse_y_axis)
        )
        settings.digital_slide_overlap_percent = cls._normalize_int_range(
            payload.get("digital_slide_overlap_percent", settings.digital_slide_overlap_percent),
            default=0,
            minimum=0,
            maximum=90,
        )
        settings.digital_slide_pixel_stride_mode = cls._normalize_digital_slide_pixel_stride_mode(
            payload.get("digital_slide_pixel_stride_mode", settings.digital_slide_pixel_stride_mode)
        )
        settings.digital_slide_x_pixel_stride = cls._normalize_int_range(
            payload.get("digital_slide_x_pixel_stride", settings.digital_slide_x_pixel_stride),
            default=1280,
            minimum=1,
            maximum=100_000,
        )
        settings.digital_slide_y_pixel_stride = cls._normalize_int_range(
            payload.get("digital_slide_y_pixel_stride", settings.digital_slide_y_pixel_stride),
            default=960,
            minimum=1,
            maximum=100_000,
        )
        settings.digital_slide_blend_width = cls._normalize_int_range(
            payload.get("digital_slide_blend_width", settings.digital_slide_blend_width),
            default=0,
            minimum=0,
            maximum=10_000,
        )
        settings.digital_slide_xy_settle_ms = cls._normalize_int_range(
            payload.get("digital_slide_xy_settle_ms", settings.digital_slide_xy_settle_ms),
            default=200,
            minimum=0,
            maximum=10_000,
        )
        settings.digital_slide_xy_post_settle_ms = cls._normalize_int_range(
            payload.get("digital_slide_xy_post_settle_ms", settings.digital_slide_xy_post_settle_ms),
            default=100,
            minimum=0,
            maximum=5000,
        )
        settings.digital_slide_z_settle_ms = cls._normalize_int_range(
            payload.get("digital_slide_z_settle_ms", settings.digital_slide_z_settle_ms),
            default=80,
            minimum=0,
            maximum=10_000,
        )
        settings.digital_slide_z_post_settle_ms = cls._normalize_int_range(
            payload.get("digital_slide_z_post_settle_ms", settings.digital_slide_z_post_settle_ms),
            default=40,
            minimum=0,
            maximum=5000,
        )
        settings.digital_slide_first_tile_extra_wait_ms = cls._normalize_int_range(
            payload.get("digital_slide_first_tile_extra_wait_ms", settings.digital_slide_first_tile_extra_wait_ms),
            default=3000,
            minimum=0,
            maximum=60_000,
        )
        settings.digital_slide_discard_frames = cls._normalize_int_range(
            payload.get("digital_slide_discard_frames", settings.digital_slide_discard_frames),
            default=2,
            minimum=0,
            maximum=20,
        )
        settings.digital_slide_focus_wheel_step = cls._normalize_int_range(
            payload.get("digital_slide_focus_wheel_step", settings.digital_slide_focus_wheel_step),
            default=1,
            minimum=1,
            maximum=10,
        )
        settings.digital_slide_dynamic_focus_overview_enabled = bool(
            payload.get(
                "digital_slide_dynamic_focus_overview_enabled",
                settings.digital_slide_dynamic_focus_overview_enabled,
            )
        )
        settings.digital_slide_smooth_navigation_enabled = bool(
            payload.get(
                "digital_slide_smooth_navigation_enabled",
                settings.digital_slide_smooth_navigation_enabled,
            )
        )
        settings.digital_slide_shift_navigation_enabled = bool(
            payload.get(
                "digital_slide_shift_navigation_enabled",
                settings.digital_slide_shift_navigation_enabled,
            )
        )
        raw_profiles = payload.get("digital_slide_profiles")
        parsed_profiles: list[DigitalSlideAcquisitionProfile] = []
        if isinstance(raw_profiles, list):
            for index, item in enumerate(raw_profiles):
                try:
                    parsed_profiles.append(DigitalSlideAcquisitionProfile.from_dict(item))
                except (TypeError, ValueError) as exc:
                    settings.load_issues.append(
                        {
                            "kind": "digital_slide_profile",
                            "index": index,
                            "message": str(exc),
                            "raw_payload": item,
                        }
                    )
        settings.digital_slide_profiles = cls._normalize_digital_slide_profiles(
            parsed_profiles,
            fallback=settings,
        )
        settings.digital_slide_active_profile_id = str(
            payload.get("digital_slide_active_profile_id", "") or ""
        ).strip()
        if settings.digital_slide_profiles:
            active_id = settings.digital_slide_active_profile_id
            if not any(
                profile.profile_id == active_id
                for profile in settings.digital_slide_profiles
            ):
                active_id = settings.digital_slide_profiles[0].profile_id
            target = next(
                profile
                for profile in settings.digital_slide_profiles
                if profile.profile_id == active_id
            )
            settings.digital_slide_active_profile_id = active_id
            cls._apply_digital_slide_profile_values(settings, target.values)
        return settings.normalized_copy()


def settings_directory() -> Path:
    if sys.platform.startswith("win"):
        base = (
            os.environ.get("LOCALAPPDATA")
            or os.environ.get("APPDATA")
            or str(Path.home() / "AppData" / "Local")
        )
        return Path(base) / "FiberDiameterMeasurement"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "FiberDiameterMeasurement"
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config_home:
        return Path(xdg_config_home) / "FiberDiameterMeasurement"
    return Path.home() / ".config" / "FiberDiameterMeasurement"


def settings_file_path() -> Path:
    return settings_directory() / "settings.json"


def legacy_area_weights_directory() -> Path:
    return settings_directory() / "area-models"


class AppSettingsIO:
    @staticmethod
    def load(path: str | Path | None = None) -> AppSettings:
        target_path = Path(path) if path is not None else settings_file_path()
        if not target_path.exists():
            return AppSettings()
        try:
            # Legacy settings may contain non-standard numeric constants. Load
            # them only so individual invalid presets can be quarantined by
            # AppSettings.from_dict; every subsequent write is strict JSON.
            payload = json.loads(target_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return AppSettings()
        if not isinstance(payload, dict):
            return AppSettings()
        return AppSettings.from_dict(payload)

    @staticmethod
    def save(settings: AppSettings, path: str | Path | None = None) -> Path:
        target_path = Path(path) if path is not None else settings_file_path()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(target_path, settings.to_dict(), ensure_ascii=False, indent=2)
        return target_path

    @staticmethod
    def replace_with_file(source_path: str | Path, path: str | Path | None = None) -> tuple[AppSettings, Path]:
        source = Path(source_path).expanduser()
        target_path = Path(path) if path is not None else settings_file_path()
        try:
            payload = json.loads(
                source.read_text(encoding="utf-8"),
                parse_constant=_reject_non_finite_json_constant,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError("设置文件不是有效的 JSON。") from exc
        if not isinstance(payload, dict):
            raise ValueError("设置文件内容不是有效的对象。")
        settings = AppSettings.from_dict(payload)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        # Persist the validated, normalized model rather than copying the raw
        # source. Python's JSON parser otherwise permits NaN/Infinity tokens
        # that our canonical settings format explicitly forbids.
        atomic_write_json(target_path, settings.to_dict(), ensure_ascii=False, indent=2)
        return settings, target_path


class DigitalSlideAcquisitionProfileIO:
    @staticmethod
    def save(
        profile: DigitalSlideAcquisitionProfile,
        path: str | Path,
        *,
        fallback: AppSettings | None = None,
    ) -> Path:
        target = Path(path).expanduser()
        base = (fallback or AppSettings()).normalized_copy()
        normalized_profiles = AppSettings._normalize_digital_slide_profiles(
            [profile],
            fallback=base,
        )
        if not normalized_profiles:
            raise ValueError("采集配置无效，无法导出。")
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            target,
            {
                "kind": DIGITAL_SLIDE_PROFILE_FILE_KIND,
                "version": DIGITAL_SLIDE_PROFILE_FILE_VERSION,
                "profile": normalized_profiles[0].to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        )
        return target

    @staticmethod
    def load(
        path: str | Path,
        *,
        fallback: AppSettings | None = None,
    ) -> DigitalSlideAcquisitionProfile:
        source = Path(path).expanduser()
        try:
            payload = json.loads(
                source.read_text(encoding="utf-8"),
                parse_constant=_reject_non_finite_json_constant,
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("采集配置文件不是有效的 JSON。") from exc
        if not isinstance(payload, dict):
            raise ValueError("采集配置文件内容不是有效的对象。")
        if payload.get("kind") != DIGITAL_SLIDE_PROFILE_FILE_KIND:
            raise ValueError("所选 JSON 不是数字切片采集配置。")
        try:
            version = int(payload.get("version", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("采集配置版本无效。") from exc
        if version != DIGITAL_SLIDE_PROFILE_FILE_VERSION:
            raise ValueError(f"暂不支持采集配置版本 {version}。")
        profile = DigitalSlideAcquisitionProfile.from_dict(payload.get("profile"))
        base = (fallback or AppSettings()).normalized_copy()
        normalized_profiles = AppSettings._normalize_digital_slide_profiles(
            [profile],
            fallback=base,
        )
        if not normalized_profiles:
            raise ValueError("采集配置缺少有效名称或参数。")
        return normalized_profiles[0]


def _reject_non_finite_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")
