from __future__ import annotations

from dataclasses import dataclass
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
class AppSettings:
    show_measurement_labels: bool = True
    measurement_label_font_family: str = "Microsoft YaHei UI"
    measurement_label_font_size: int = 14
    measurement_label_color: str = "#FFFFFF"
    measurement_endpoint_style: str = MeasurementEndpointStyle.CIRCLE
    default_measurement_color: str = "#E0FBFC"
    open_image_view_mode: str = OpenImageViewMode.DEFAULT
    scale_overlay_placement_mode: str = ScaleOverlayPlacementMode.BOTTOM_LEFT
    text_font_family: str = "Microsoft YaHei UI"
    text_font_size: int = 18
    text_color: str = "#F7F4EA"

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "show_measurement_labels": self.show_measurement_labels,
            "measurement_label_font_family": self.measurement_label_font_family,
            "measurement_label_font_size": self.measurement_label_font_size,
            "measurement_label_color": self.measurement_label_color,
            "measurement_endpoint_style": self.measurement_endpoint_style,
            "default_measurement_color": self.default_measurement_color,
            "open_image_view_mode": self.open_image_view_mode,
            "scale_overlay_placement_mode": self.scale_overlay_placement_mode,
            "text_font_family": self.text_font_family,
            "text_font_size": self.text_font_size,
            "text_color": self.text_color,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "AppSettings":
        settings = cls()
        settings.show_measurement_labels = bool(payload.get("show_measurement_labels", settings.show_measurement_labels))
        settings.measurement_label_font_family = str(payload.get("measurement_label_font_family", settings.measurement_label_font_family))
        settings.measurement_label_font_size = int(payload.get("measurement_label_font_size", settings.measurement_label_font_size))
        settings.measurement_label_color = str(payload.get("measurement_label_color", settings.measurement_label_color))
        settings.measurement_endpoint_style = str(payload.get("measurement_endpoint_style", settings.measurement_endpoint_style))
        settings.default_measurement_color = str(payload.get("default_measurement_color", settings.default_measurement_color))
        settings.open_image_view_mode = str(payload.get("open_image_view_mode", settings.open_image_view_mode))
        settings.scale_overlay_placement_mode = str(payload.get("scale_overlay_placement_mode", settings.scale_overlay_placement_mode))
        settings.text_font_family = str(payload.get("text_font_family", settings.text_font_family))
        settings.text_font_size = int(payload.get("text_font_size", settings.text_font_size))
        settings.text_color = str(payload.get("text_color", settings.text_color))
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
