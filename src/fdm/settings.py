from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import json
import os
import sys

from fdm.models import CalibrationPreset


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


class FocusStackProfile:
    SHARP = "sharp"
    BALANCED = "balanced"
    SOFT = "soft"


class MagicSegmentModelVariant:
    EDGE_SAM = "edge_sam"
    EDGE_SAM_3X = "edge_sam_3x"


class ComplexMagicSegmentModelVariant:
    LIGHT_HQ_SAM = "light_hq_sam"
    EFFICIENTSAM_S = "efficientsam_s"


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


class RawRecordDataSource:
    DIAMETER_RESULT = "diameter_result"
    AREA_RESULT = "area_result"
    MEASUREMENT_FIELD = "measurement_field"


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
    direction: str = RawRecordExportDirection.VERTICAL

    def normalized_copy(self) -> "RawRecordExportRule":
        return RawRecordExportRule(
            data_source=self._normalize_data_source(self.data_source),
            field_name=str(self.field_name or "结果").strip() or "结果",
            measurement_filter=self._normalize_measurement_filter(self.measurement_filter),
            sheet_name=str(self.sheet_name or "Sheet1").strip() or "Sheet1",
            start_cell=str(self.start_cell or "B2").strip().upper() or "B2",
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
            direction=cls._normalize_direction(str(payload.get("direction", RawRecordExportDirection.VERTICAL))),
        ).normalized_copy()

    @staticmethod
    def _normalize_data_source(value: str | None) -> str:
        token = str(value or "").strip()
        if token in {
            RawRecordDataSource.DIAMETER_RESULT,
            RawRecordDataSource.AREA_RESULT,
            RawRecordDataSource.MEASUREMENT_FIELD,
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
    return [
        AreaModelMapping(model_name="粘纤-莱赛尔", model_file="b_v1_1.3.pth"),
        AreaModelMapping(model_name="棉-粘纤", model_file="b_cv_1.3.pth"),
        AreaModelMapping(model_name="棉-莱赛尔", model_file="b_c1_1.3.pth"),
        AreaModelMapping(model_name="棉-莫代尔", model_file="b_cm_1.3.pth"),
        AreaModelMapping(model_name="棉-再生纤维素纤维", model_file="b_cc_1.3.pth"),
        AreaModelMapping(model_name="棉-粘-莱-莫", model_file="b_cvlm_1.3.pth"),
    ]


@dataclass(slots=True)
class AppSettings:
    theme_mode: str = AppThemeMode.DARK
    show_measurement_labels: bool = True
    measurement_label_font_family: str = "Microsoft YaHei UI"
    measurement_label_font_size: int = 14
    measurement_label_color: str = "#00FF00"
    measurement_label_decimals: int = 2
    measurement_label_parallel_to_line: bool = False
    measurement_label_background_enabled: bool = False
    measurement_endpoint_style: str = MeasurementEndpointStyle.BAR
    default_measurement_color: str = "#E0FBFC"
    open_image_view_mode: str = OpenImageViewMode.FIT
    scale_overlay_placement_mode: str = ScaleOverlayPlacementMode.BOTTOM_RIGHT
    scale_overlay_style: str = ScaleOverlayStyle.TICKS
    scale_overlay_length_value: float = 50.0
    scale_overlay_color: str = "#FF0000"
    scale_overlay_text_color: str = "#FF0000"
    scale_overlay_font_family: str = "Microsoft YaHei UI"
    scale_overlay_font_size: int = 18
    text_font_family: str = "Microsoft YaHei UI"
    text_font_size: int = 18
    text_color: str = "#F7F4EA"
    overlay_line_color: str = "#F7F4EA"
    overlay_line_width: float = 2.5
    focus_stack_profile: str = FocusStackProfile.BALANCED
    focus_stack_sharpen_strength: int = 35
    magic_segment_model_variant: str = MagicSegmentModelVariant.EDGE_SAM_3X
    magic_segment_fill_draft_holes_enabled: bool = False
    magic_segment_standard_roi_enabled: bool = False
    fiber_quick_roi_enabled: bool = True
    fiber_quick_edge_trim_enabled: bool = True
    fiber_quick_line_extension_px: float = 0.0
    main_window_geometry: str = ""
    main_window_is_maximized: bool = False
    recent_export_dir: str = ""
    recent_project_dir: str = ""
    area_model_mappings: list[AreaModelMapping] = field(default_factory=default_area_model_mappings)
    area_weights_dir: str = field(default_factory=default_area_weights_directory)
    area_vendor_root: str = field(default_factory=default_area_vendor_root)
    area_worker_python: str = field(default_factory=default_area_worker_python)
    calibration_presets: list[CalibrationPreset] = field(default_factory=list)
    selected_capture_device_id: str = ""
    raw_record_templates: list[RawRecordTemplate] = field(default_factory=list)
    last_raw_record_template_path: str = ""

    def normalized_copy(self) -> "AppSettings":
        normalized = replace(self)
        normalized.theme_mode = normalize_theme_mode(self.theme_mode)
        normalized.measurement_label_font_size = self._normalize_font_size(self.measurement_label_font_size, minimum=8, maximum=96)
        normalized.measurement_label_decimals = self._normalize_measurement_label_decimals(self.measurement_label_decimals)
        normalized.measurement_endpoint_style = self._normalize_measurement_endpoint_style(self.measurement_endpoint_style)
        normalized.open_image_view_mode = self._normalize_open_image_view_mode(self.open_image_view_mode)
        normalized.scale_overlay_placement_mode = self._normalize_scale_overlay_placement_mode(self.scale_overlay_placement_mode)
        normalized.scale_overlay_style = self._normalize_scale_overlay_style(self.scale_overlay_style)
        normalized.scale_overlay_length_value = self._normalize_scale_overlay_length_value(self.scale_overlay_length_value)
        normalized.scale_overlay_font_size = self._normalize_font_size(self.scale_overlay_font_size, minimum=8, maximum=96)
        normalized.text_font_size = self._normalize_font_size(self.text_font_size, minimum=8, maximum=144)
        normalized.overlay_line_width = self._normalize_overlay_line_width(self.overlay_line_width)
        normalized.focus_stack_profile = self._normalize_focus_stack_profile(self.focus_stack_profile)
        normalized.focus_stack_sharpen_strength = self._normalize_focus_stack_sharpen_strength(self.focus_stack_sharpen_strength)
        normalized.magic_segment_model_variant = self._normalize_magic_segment_model_variant(self.magic_segment_model_variant)
        normalized.fiber_quick_line_extension_px = self._normalize_fiber_quick_line_extension_px(self.fiber_quick_line_extension_px)
        normalized.recent_export_dir = self._normalize_recent_directory(self.recent_export_dir)
        normalized.recent_project_dir = self._normalize_recent_directory(self.recent_project_dir)
        normalized.area_weights_dir = self._normalize_weights_dir(self.area_weights_dir)
        normalized.area_vendor_root = self._normalize_vendor_root(self.area_vendor_root)
        normalized.area_worker_python = self._normalize_worker_program(self.area_worker_python)
        normalized.raw_record_templates = self._normalize_raw_record_templates(self.raw_record_templates)
        normalized.last_raw_record_template_path = normalize_raw_record_template_path(self.last_raw_record_template_path)
        return normalized

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

    def to_dict(self) -> dict[str, object]:
        normalized = self.normalized_copy()
        return {
            "version": 1,
            "theme_mode": normalized.theme_mode,
            "show_measurement_labels": normalized.show_measurement_labels,
            "measurement_label_font_family": normalized.measurement_label_font_family,
            "measurement_label_font_size": normalized.measurement_label_font_size,
            "measurement_label_color": normalized.measurement_label_color,
            "measurement_label_decimals": normalized.measurement_label_decimals,
            "measurement_label_parallel_to_line": normalized.measurement_label_parallel_to_line,
            "measurement_label_background_enabled": normalized.measurement_label_background_enabled,
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
            "overlay_line_color": normalized.overlay_line_color,
            "overlay_line_width": normalized.overlay_line_width,
            "focus_stack_profile": normalized.focus_stack_profile,
            "focus_stack_sharpen_strength": normalized.focus_stack_sharpen_strength,
            "magic_segment_model_variant": normalized.magic_segment_model_variant,
            "magic_segment_fill_draft_holes_enabled": normalized.magic_segment_fill_draft_holes_enabled,
            "magic_segment_standard_roi_enabled": normalized.magic_segment_standard_roi_enabled,
            "fiber_quick_roi_enabled": normalized.fiber_quick_roi_enabled,
            "fiber_quick_edge_trim_enabled": normalized.fiber_quick_edge_trim_enabled,
            "fiber_quick_line_extension_px": normalized.fiber_quick_line_extension_px,
            "main_window_geometry": normalized.main_window_geometry,
            "main_window_is_maximized": normalized.main_window_is_maximized,
            "recent_export_dir": normalized.recent_export_dir,
            "recent_project_dir": normalized.recent_project_dir,
            "area_model_mappings": [item.to_dict() for item in normalized.area_model_mappings],
            "area_weights_dir": normalized.area_weights_dir,
            "area_vendor_root": normalized.area_vendor_root,
            "area_worker_python": normalized.area_worker_python,
            "calibration_presets": [preset.to_dict() for preset in normalized.calibration_presets],
            "selected_capture_device_id": normalized.selected_capture_device_id,
            "raw_record_templates": [template.to_dict() for template in normalized.raw_record_templates],
            "last_raw_record_template_path": normalized.last_raw_record_template_path,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "AppSettings":
        settings = cls()
        settings.theme_mode = normalize_theme_mode(payload.get("theme_mode", settings.theme_mode))
        settings.show_measurement_labels = bool(payload.get("show_measurement_labels", settings.show_measurement_labels))
        settings.measurement_label_font_family = str(payload.get("measurement_label_font_family", settings.measurement_label_font_family))
        settings.measurement_label_font_size = cls._normalize_font_size(
            payload.get("measurement_label_font_size", settings.measurement_label_font_size),
            minimum=8,
            maximum=96,
        )
        settings.measurement_label_color = str(payload.get("measurement_label_color", settings.measurement_label_color))
        settings.measurement_label_decimals = cls._normalize_measurement_label_decimals(
            payload.get("measurement_label_decimals", settings.measurement_label_decimals)
        )
        settings.measurement_label_parallel_to_line = bool(payload.get("measurement_label_parallel_to_line", settings.measurement_label_parallel_to_line))
        settings.measurement_label_background_enabled = bool(payload.get("measurement_label_background_enabled", settings.measurement_label_background_enabled))
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
        settings.overlay_line_color = str(payload.get("overlay_line_color", settings.overlay_line_color))
        settings.overlay_line_width = cls._normalize_overlay_line_width(
            payload.get("overlay_line_width", settings.overlay_line_width)
        )
        settings.focus_stack_profile = cls._normalize_focus_stack_profile(payload.get("focus_stack_profile", settings.focus_stack_profile))
        settings.focus_stack_sharpen_strength = cls._normalize_focus_stack_sharpen_strength(
            payload.get("focus_stack_sharpen_strength", settings.focus_stack_sharpen_strength)
        )
        settings.magic_segment_model_variant = cls._normalize_magic_segment_model_variant(
            payload.get("magic_segment_model_variant", settings.magic_segment_model_variant)
        )
        settings.magic_segment_fill_draft_holes_enabled = bool(
            payload.get(
                "magic_segment_fill_draft_holes_enabled",
                settings.magic_segment_fill_draft_holes_enabled,
            )
        )
        settings.magic_segment_standard_roi_enabled = bool(
            payload.get(
                "magic_segment_standard_roi_enabled",
                settings.magic_segment_standard_roi_enabled,
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
        presets = payload.get("calibration_presets", None)
        if isinstance(presets, list):
            settings.calibration_presets = [
                CalibrationPreset.from_dict(item)
                for item in presets
                if isinstance(item, dict) and str(item.get("name", "")).strip()
            ]
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
        return settings


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
        target_path.write_text(
            json.dumps(settings.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return target_path
