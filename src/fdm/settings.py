from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import json
import os
import sys


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
    show_measurement_labels: bool = True
    measurement_label_font_family: str = "Microsoft YaHei UI"
    measurement_label_font_size: int = 14
    measurement_label_color: str = "#FFFFFF"
    measurement_label_decimals: int = 4
    measurement_label_parallel_to_line: bool = False
    measurement_label_background_enabled: bool = True
    measurement_endpoint_style: str = MeasurementEndpointStyle.CIRCLE
    default_measurement_color: str = "#E0FBFC"
    open_image_view_mode: str = OpenImageViewMode.DEFAULT
    scale_overlay_placement_mode: str = ScaleOverlayPlacementMode.BOTTOM_LEFT
    text_font_family: str = "Microsoft YaHei UI"
    text_font_size: int = 18
    text_color: str = "#F7F4EA"
    area_model_mappings: list[AreaModelMapping] = field(default_factory=default_area_model_mappings)
    area_weights_dir: str = field(default_factory=default_area_weights_directory)
    area_vendor_root: str = field(default_factory=default_area_vendor_root)
    area_worker_python: str = field(default_factory=default_area_worker_python)

    def normalized_copy(self) -> "AppSettings":
        normalized = replace(self)
        normalized.area_weights_dir = self._normalize_weights_dir(self.area_weights_dir)
        normalized.area_vendor_root = self._normalize_vendor_root(self.area_vendor_root)
        normalized.area_worker_python = self._normalize_worker_program(self.area_worker_python)
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

    def to_dict(self) -> dict[str, object]:
        normalized = self.normalized_copy()
        return {
            "version": 1,
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
            "text_font_family": normalized.text_font_family,
            "text_font_size": normalized.text_font_size,
            "text_color": normalized.text_color,
            "area_model_mappings": [item.to_dict() for item in normalized.area_model_mappings],
            "area_weights_dir": normalized.area_weights_dir,
            "area_vendor_root": normalized.area_vendor_root,
            "area_worker_python": normalized.area_worker_python,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "AppSettings":
        settings = cls()
        settings.show_measurement_labels = bool(payload.get("show_measurement_labels", settings.show_measurement_labels))
        settings.measurement_label_font_family = str(payload.get("measurement_label_font_family", settings.measurement_label_font_family))
        settings.measurement_label_font_size = int(payload.get("measurement_label_font_size", settings.measurement_label_font_size))
        settings.measurement_label_color = str(payload.get("measurement_label_color", settings.measurement_label_color))
        settings.measurement_label_decimals = int(payload.get("measurement_label_decimals", settings.measurement_label_decimals))
        settings.measurement_label_parallel_to_line = bool(payload.get("measurement_label_parallel_to_line", settings.measurement_label_parallel_to_line))
        settings.measurement_label_background_enabled = bool(payload.get("measurement_label_background_enabled", settings.measurement_label_background_enabled))
        settings.measurement_endpoint_style = str(payload.get("measurement_endpoint_style", settings.measurement_endpoint_style))
        settings.default_measurement_color = str(payload.get("default_measurement_color", settings.default_measurement_color))
        settings.open_image_view_mode = str(payload.get("open_image_view_mode", settings.open_image_view_mode))
        settings.scale_overlay_placement_mode = str(payload.get("scale_overlay_placement_mode", settings.scale_overlay_placement_mode))
        settings.text_font_family = str(payload.get("text_font_family", settings.text_font_family))
        settings.text_font_size = int(payload.get("text_font_size", settings.text_font_size))
        settings.text_color = str(payload.get("text_color", settings.text_color))
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
