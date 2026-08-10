from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from string import Formatter
from typing import Callable
import json
import math
import os
import re
import sys

from fdm.atomic_io import atomic_write_json
from fdm.services.screenshot_capture import CaptureMode, CaptureRect


SCREENSHOT_SETTINGS_SCHEMA_VERSION = 1
SCREENSHOT_SETTINGS_FILE_NAME = "screenshot-settings.json"
SCREENSHOT_SETTINGS_LOCK_TIMEOUT_MS = 5_000
DEFAULT_FILENAME_TEMPLATE = "Screenshot_{date}_{time}"
_ALLOWED_FILENAME_FIELDS = frozenset({"date", "time", "datetime", "mode", "counter"})
_MISSING = object()


class UnsupportedScreenshotSettingsVersion(ValueError):
    """Raised when settings were written by a newer companion application."""


class ImageFormat(str, Enum):
    PNG = "png"
    JPEG = "jpeg"
    WEBP = "webp"

    @classmethod
    def parse(cls, value: object, *, default: "ImageFormat" | None = None) -> "ImageFormat":
        if isinstance(value, cls):
            return value
        token = str(value or "").strip().lower().lstrip(".")
        if token == "jpg":
            token = cls.JPEG.value
        try:
            return cls(token)
        except ValueError:
            if default is not None:
                return default
            raise

    @property
    def suffix(self) -> str:
        return ".jpg" if self is ImageFormat.JPEG else f".{self.value}"

    @property
    def qt_format(self) -> bytes:
        return {
            ImageFormat.PNG: b"PNG",
            ImageFormat.JPEG: b"JPEG",
            ImageFormat.WEBP: b"WEBP",
        }[self]


class CollisionPolicy(str, Enum):
    INCREMENT = "increment"
    OVERWRITE = "overwrite"
    FAIL = "fail"

    @classmethod
    def parse(
        cls,
        value: object,
        *,
        default: "CollisionPolicy" | None = None,
    ) -> "CollisionPolicy":
        if isinstance(value, cls):
            return value
        token = str(value or "").strip().lower().replace("-", "_")
        aliases = {"rename": cls.INCREMENT, "error": cls.FAIL}
        if token in aliases:
            return aliases[token]
        try:
            return cls(token)
        except ValueError:
            if default is not None:
                return default
            raise


class AfterCaptureTask(str, Enum):
    SAVE = "save"
    COPY_CLIPBOARD = "copy_clipboard"

    @classmethod
    def parse(cls, value: object) -> "AfterCaptureTask":
        if isinstance(value, cls):
            return value
        token = str(value or "").strip().lower().replace("-", "_")
        aliases = {
            "clipboard": cls.COPY_CLIPBOARD,
            "copy": cls.COPY_CLIPBOARD,
            "copy_to_clipboard": cls.COPY_CLIPBOARD,
        }
        if token in aliases:
            return aliases[token]
        return cls(token)


@dataclass(frozen=True, slots=True)
class HotkeyBinding:
    sequence: str = ""
    enabled: bool = True

    def normalized(self) -> "HotkeyBinding":
        sequence = "+".join(
            part.strip()
            for part in str(self.sequence or "").split("+")
            if part.strip()
        )
        sequence = "+".join(
            "Print" if part.casefold() in {"printscreen", "prtsc", "prtscn"} else part
            for part in sequence.split("+")
        )
        return HotkeyBinding(sequence=sequence, enabled=bool(self.enabled and sequence))

    def to_dict(self) -> dict[str, object]:
        normalized = self.normalized()
        return {"sequence": normalized.sequence, "enabled": normalized.enabled}

    @classmethod
    def from_value(
        cls,
        value: object,
        *,
        default: "HotkeyBinding" | None = None,
    ) -> "HotkeyBinding":
        fallback = default or cls()
        if isinstance(value, cls):
            return value.normalized()
        if isinstance(value, str):
            return cls(sequence=value, enabled=True).normalized()
        if not isinstance(value, dict):
            return fallback.normalized()
        return cls(
            sequence=str(value.get("sequence", fallback.sequence) or ""),
            enabled=_normalize_bool(value.get("enabled"), fallback.enabled),
        ).normalized()


def default_hotkeys() -> dict[CaptureMode, HotkeyBinding]:
    """Return conflict-aware defaults; none are registered while ``enabled`` is false."""

    return {
        CaptureMode.REGION: HotkeyBinding("Print"),
        CaptureMode.WINDOW: HotkeyBinding("Alt+Print"),
        CaptureMode.FULL_SCREEN: HotkeyBinding("Ctrl+Print"),
        CaptureMode.LAST_REGION: HotkeyBinding("Shift+Print"),
        CaptureMode.CU5: HotkeyBinding("Ctrl+Shift+Print"),
    }


@dataclass(slots=True)
class ScreenshotSettings:
    """Version-one persistent settings owned by the screenshot companion.

    ``last_region`` always uses native desktop pixels.  It may therefore have a
    negative origin on a display positioned left or above the primary display.
    """

    enabled: bool = False
    autostart: bool = False
    output_directory: str = field(
        default_factory=lambda: str(Path.home() / "Pictures" / "Screenshots")
    )
    filename_template: str = DEFAULT_FILENAME_TEMPLATE
    image_format: ImageFormat = ImageFormat.PNG
    png_compression: int = 6
    jpeg_quality: int = 92
    webp_quality: int = 90
    collision_policy: CollisionPolicy = CollisionPolicy.INCREMENT
    after_capture_tasks: tuple[AfterCaptureTask, ...] = (AfterCaptureTask.SAVE,)
    delay_ms: int = 0
    include_cursor: bool = False
    show_editor: bool = False
    notification: bool = True
    hotkeys: dict[CaptureMode, HotkeyBinding] = field(default_factory=default_hotkeys)
    cu5_selector: dict[str, object] = field(default_factory=dict)
    cu5_diagnostics_enabled: bool = False
    last_region: CaptureRect | None = None

    @property
    def schema_version(self) -> int:
        return SCREENSHOT_SETTINGS_SCHEMA_VERSION

    @property
    def background_resident(self) -> bool:
        return self.enabled

    @background_resident.setter
    def background_resident(self, value: object) -> None:
        self.enabled = _normalize_bool(value, False)

    @property
    def notifications_enabled(self) -> bool:
        return self.notification

    @notifications_enabled.setter
    def notifications_enabled(self, value: object) -> None:
        self.notification = _normalize_bool(value, True)

    def normalized(self) -> "ScreenshotSettings":
        defaults = ScreenshotSettings()
        output_directory = str(self.output_directory or "").strip()
        if not output_directory:
            output_directory = defaults.output_directory
        filename_template = _normalize_filename_template(self.filename_template)
        image_format = ImageFormat.parse(self.image_format, default=ImageFormat.PNG)
        collision_policy = CollisionPolicy.parse(
            self.collision_policy,
            default=CollisionPolicy.INCREMENT,
        )
        tasks = _normalize_after_capture_tasks(self.after_capture_tasks)
        hotkeys = _normalize_hotkeys(self.hotkeys)
        last_region = _normalize_capture_rect(self.last_region)
        selector = _normalize_json_object(self.cu5_selector)
        return ScreenshotSettings(
            enabled=_normalize_bool(self.enabled, False),
            autostart=_normalize_bool(self.autostart, False),
            output_directory=output_directory,
            filename_template=filename_template,
            image_format=image_format,
            png_compression=_bounded_int(self.png_compression, 0, 9, 6),
            jpeg_quality=_bounded_int(self.jpeg_quality, 1, 100, 92),
            webp_quality=_bounded_int(self.webp_quality, 1, 100, 90),
            collision_policy=collision_policy,
            after_capture_tasks=tasks,
            delay_ms=_bounded_int(self.delay_ms, 0, 60_000, 0),
            include_cursor=_normalize_bool(self.include_cursor, False),
            show_editor=_normalize_bool(self.show_editor, False),
            notification=_normalize_bool(self.notification, True),
            hotkeys=hotkeys,
            cu5_selector=selector,
            cu5_diagnostics_enabled=_normalize_bool(self.cu5_diagnostics_enabled, False),
            last_region=last_region,
        )

    def to_dict(self) -> dict[str, object]:
        settings = self.normalized()
        last_region = None
        if settings.last_region is not None:
            last_region = {
                "x": settings.last_region.x,
                "y": settings.last_region.y,
                "width": settings.last_region.width,
                "height": settings.last_region.height,
                "coordinate_space": "physical_pixels",
            }
        return {
            "schema_version": SCREENSHOT_SETTINGS_SCHEMA_VERSION,
            "enabled": settings.enabled,
            "autostart": settings.autostart,
            "output_directory": settings.output_directory,
            "filename_template": settings.filename_template,
            "image_format": settings.image_format.value,
            "png_compression": settings.png_compression,
            "jpeg_quality": settings.jpeg_quality,
            "webp_quality": settings.webp_quality,
            "collision_policy": settings.collision_policy.value,
            "after_capture_tasks": [item.value for item in settings.after_capture_tasks],
            "delay_ms": settings.delay_ms,
            "include_cursor": settings.include_cursor,
            "show_editor": settings.show_editor,
            "notification": settings.notification,
            "hotkeys": {
                mode.value: binding.to_dict()
                for mode, binding in settings.hotkeys.items()
            },
            "cu5_selector": settings.cu5_selector,
            "cu5_diagnostics_enabled": settings.cu5_diagnostics_enabled,
            "last_region": last_region,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "ScreenshotSettings":
        if not isinstance(payload, dict):
            return cls()
        raw_version = payload.get("schema_version", payload.get("version", 1))
        version = _bounded_int(raw_version, 0, 1_000_000, 1)
        if version > SCREENSHOT_SETTINGS_SCHEMA_VERSION:
            raise UnsupportedScreenshotSettingsVersion(
                f"unsupported screenshot settings version: {version}"
            )
        defaults = cls()
        notification_value = payload.get(
            "notification",
            payload.get("notifications_enabled", defaults.notification),
        )
        enabled_value = payload.get(
            "enabled",
            payload.get("background_resident", defaults.enabled),
        )
        return cls(
            enabled=_normalize_bool(enabled_value, defaults.enabled),
            autostart=_normalize_bool(payload.get("autostart"), defaults.autostart),
            output_directory=str(
                payload.get("output_directory", defaults.output_directory) or ""
            ),
            filename_template=str(
                payload.get("filename_template", defaults.filename_template) or ""
            ),
            image_format=ImageFormat.parse(
                payload.get("image_format"), default=defaults.image_format
            ),
            png_compression=_bounded_int(
                payload.get("png_compression"), 0, 9, defaults.png_compression
            ),
            jpeg_quality=_bounded_int(
                payload.get("jpeg_quality"), 1, 100, defaults.jpeg_quality
            ),
            webp_quality=_bounded_int(
                payload.get("webp_quality"), 1, 100, defaults.webp_quality
            ),
            collision_policy=CollisionPolicy.parse(
                payload.get("collision_policy"), default=defaults.collision_policy
            ),
            after_capture_tasks=_normalize_after_capture_tasks(
                payload.get("after_capture_tasks", defaults.after_capture_tasks)
            ),
            delay_ms=_bounded_int(payload.get("delay_ms"), 0, 60_000, 0),
            include_cursor=_normalize_bool(payload.get("include_cursor"), False),
            show_editor=_normalize_bool(payload.get("show_editor"), False),
            notification=_normalize_bool(notification_value, True),
            hotkeys=_normalize_hotkeys(payload.get("hotkeys")),
            cu5_selector=_normalize_json_object(payload.get("cu5_selector")),
            cu5_diagnostics_enabled=_normalize_bool(
                payload.get("cu5_diagnostics_enabled"), False
            ),
            last_region=_normalize_capture_rect(payload.get("last_region")),
        ).normalized()


def screenshot_settings_directory() -> Path:
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


def screenshot_settings_file_path() -> Path:
    return screenshot_settings_directory() / SCREENSHOT_SETTINGS_FILE_NAME


# A short alias is convenient in the companion process without colliding with
# ``fdm.settings.settings_file_path`` at the module level.
settings_file_path = screenshot_settings_file_path


class ScreenshotSettingsIO:
    @staticmethod
    def _target(path: str | Path | None = None) -> Path:
        return Path(path) if path is not None else screenshot_settings_file_path()

    @staticmethod
    def _load_unlocked(target: Path) -> ScreenshotSettings:
        if not target.exists():
            return ScreenshotSettings()
        try:
            payload = json.loads(
                target.read_text(encoding="utf-8"),
                parse_constant=_reject_non_finite_json_constant,
            )
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return ScreenshotSettings()
        return ScreenshotSettings.from_dict(payload)

    @staticmethod
    def _save_unlocked(settings: ScreenshotSettings, target: Path) -> Path:
        atomic_write_json(target, settings.to_dict(), ensure_ascii=False, indent=2)
        return target

    @staticmethod
    def _lock(target: Path):
        # QLockFile uses an adjacent, process-aware lock record and can recover
        # one left by a crashed process.  The settings file itself remains an
        # atomic-replace target, so readers never observe a partial JSON file.
        from PySide6.QtCore import QLockFile

        lock_path = target.with_name(f".{target.name}.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock = QLockFile(str(lock_path))
        lock.setStaleLockTime(30_000)
        if not lock.tryLock(SCREENSHOT_SETTINGS_LOCK_TIMEOUT_MS):
            raise OSError(
                "无法取得截图设置跨进程写入锁："
                f"{getattr(lock.error(), 'name', str(lock.error()))}"
            )
        return lock

    @staticmethod
    def load(path: str | Path | None = None) -> ScreenshotSettings:
        return ScreenshotSettingsIO._load_unlocked(ScreenshotSettingsIO._target(path))

    @staticmethod
    def save(
        settings: ScreenshotSettings,
        path: str | Path | None = None,
    ) -> Path:
        if not isinstance(settings, ScreenshotSettings):
            raise TypeError("settings must be a ScreenshotSettings instance")
        target = ScreenshotSettingsIO._target(path)
        lock = ScreenshotSettingsIO._lock(target)
        try:
            # Validate an existing file before replacement so even low-level
            # callers cannot accidentally downgrade a future schema.
            ScreenshotSettingsIO._load_unlocked(target)
            return ScreenshotSettingsIO._save_unlocked(settings, target)
        finally:
            lock.unlock()

    @staticmethod
    def update(
        mutator: Callable[[ScreenshotSettings], ScreenshotSettings],
        path: str | Path | None = None,
        *,
        allow_unsupported_replace: bool = False,
    ) -> ScreenshotSettings:
        """Atomically read, merge and persist settings across app processes.

        A future schema is rejected before ``mutator`` runs unless the caller
        has obtained explicit user consent to replace it.
        """

        if not callable(mutator):
            raise TypeError("mutator must be callable")
        target = ScreenshotSettingsIO._target(path)
        lock = ScreenshotSettingsIO._lock(target)
        try:
            try:
                current = ScreenshotSettingsIO._load_unlocked(target).normalized()
            except UnsupportedScreenshotSettingsVersion:
                if not allow_unsupported_replace:
                    raise
                current = ScreenshotSettings().normalized()
            updated = mutator(current)
            if not isinstance(updated, ScreenshotSettings):
                raise TypeError("settings mutator must return ScreenshotSettings")
            normalized = updated.normalized()
            ScreenshotSettingsIO._save_unlocked(normalized, target)
            return normalized
        finally:
            lock.unlock()


PhysicalPixelRect = CaptureRect


def _normalize_filename_template(value: object) -> str:
    template = str(value or "").strip()
    if not template or len(template) > 240 or "/" in template or "\\" in template:
        return DEFAULT_FILENAME_TEMPLATE
    try:
        for _literal, field_name, _format_spec, conversion in Formatter().parse(template):
            if field_name is None:
                continue
            if field_name not in _ALLOWED_FILENAME_FIELDS or conversion is not None:
                return DEFAULT_FILENAME_TEMPLATE
            if _format_spec and not (
                field_name == "counter"
                and re.fullmatch(r"0?[1-6]?d", _format_spec) is not None
            ):
                return DEFAULT_FILENAME_TEMPLATE
    except ValueError:
        return DEFAULT_FILENAME_TEMPLATE
    return template


def _normalize_hotkeys(value: object) -> dict[CaptureMode, HotkeyBinding]:
    result = default_hotkeys()
    if not isinstance(value, dict):
        return result
    for raw_mode, raw_binding in value.items():
        try:
            mode = CaptureMode.parse(raw_mode)
        except ValueError:
            continue
        default = result.get(mode, HotkeyBinding("", False))
        result[mode] = HotkeyBinding.from_value(raw_binding, default=default)
    return result


def _normalize_after_capture_tasks(value: object) -> tuple[AfterCaptureTask, ...]:
    if isinstance(value, (str, AfterCaptureTask)):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = [AfterCaptureTask.SAVE]
    result: list[AfterCaptureTask] = []
    for item in values:
        try:
            task = AfterCaptureTask.parse(item)
        except (TypeError, ValueError):
            continue
        if task not in result:
            result.append(task)
    return tuple(result) or (AfterCaptureTask.SAVE,)


def _normalize_capture_rect(value: object) -> CaptureRect | None:
    if isinstance(value, CaptureRect):
        rect = value.normalized()
        return rect if rect.valid else None
    if not isinstance(value, dict):
        return None
    try:
        rect = CaptureRect(
            int(value.get("x", 0)),
            int(value.get("y", 0)),
            int(value.get("width", 0)),
            int(value.get("height", 0)),
        ).normalized()
    except (TypeError, ValueError, OverflowError):
        return None
    return rect if rect.valid else None


def _normalize_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "on", "enabled"}:
            return True
        if token in {"0", "false", "no", "off", "disabled"}:
            return False
    return default


def _bounded_int(value: object, minimum: int, maximum: int, default: int) -> int:
    try:
        if isinstance(value, bool):
            raise ValueError
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return max(minimum, min(maximum, number))


def _normalize_json_object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, object] = {}
    for key, item in value.items():
        normalized = _normalize_json_value(item, depth=0)
        if normalized is not _MISSING:
            result[str(key)] = normalized
    return result


def _normalize_json_value(value: object, *, depth: int) -> object:
    if depth > 12:
        return _MISSING
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else _MISSING
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            normalized = _normalize_json_value(item, depth=depth + 1)
            if normalized is not _MISSING:
                result.append(normalized)
        return result
    if isinstance(value, dict):
        result: dict[str, object] = {}
        for key, item in value.items():
            normalized = _normalize_json_value(item, depth=depth + 1)
            if normalized is not _MISSING:
                result[str(key)] = normalized
        return result
    return _MISSING


def _reject_non_finite_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


__all__ = [
    "AfterCaptureTask",
    "CaptureMode",
    "CollisionPolicy",
    "DEFAULT_FILENAME_TEMPLATE",
    "HotkeyBinding",
    "ImageFormat",
    "PhysicalPixelRect",
    "SCREENSHOT_SETTINGS_FILE_NAME",
    "SCREENSHOT_SETTINGS_LOCK_TIMEOUT_MS",
    "SCREENSHOT_SETTINGS_SCHEMA_VERSION",
    "ScreenshotSettings",
    "ScreenshotSettingsIO",
    "UnsupportedScreenshotSettingsVersion",
    "default_hotkeys",
    "screenshot_settings_directory",
    "screenshot_settings_file_path",
    "settings_file_path",
]
