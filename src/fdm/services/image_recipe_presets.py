"""Versioned, atomic storage for reusable image-processing recipes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import StrEnum
import json
from pathlib import Path
import unicodedata

from fdm.atomic_io import atomic_write_json
from fdm.image_processing_models import ImageProcessingRecipe
from fdm.services.image_processing import ImageOperation
from fdm.settings import settings_file_path


IMAGE_RECIPE_PRESET_FORMAT = "fdm.image-processing-recipes"
IMAGE_RECIPE_PRESET_SCHEMA_VERSION = 1
IMAGE_RECIPE_PRESET_FILENAME = "image-processing-recipes.json"


class ImageRecipePresetErrorCode(StrEnum):
    INVALID_FILE = "invalid_file"
    UNSUPPORTED_VERSION = "unsupported_version"
    INVALID_PRESET = "invalid_preset"
    UNKNOWN_OPERATION = "unknown_operation"
    UNSUPPORTED_OPERATION = "unsupported_operation"
    DUPLICATE_NAME = "duplicate_name"
    WRITE_FAILED = "write_failed"


class ImageRecipePresetError(RuntimeError):
    def __init__(
        self,
        code: ImageRecipePresetErrorCode,
        message: str,
    ) -> None:
        super().__init__(str(message))
        self.code = code
        self.message = str(message)


@dataclass(frozen=True, slots=True)
class ImageRecipePreset:
    name: str
    recipe: ImageProcessingRecipe
    created_at: str
    updated_at: str

    def __post_init__(self) -> None:
        name = _normalize_preset_name(self.name)
        _validate_recipe(self.recipe)
        created_at = _normalize_timestamp(self.created_at, "created_at")
        updated_at = _normalize_timestamp(self.updated_at, "updated_at")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "updated_at", updated_at)

    @classmethod
    def create(
        cls,
        name: str,
        recipe: ImageProcessingRecipe,
        *,
        timestamp: str | None = None,
    ) -> "ImageRecipePreset":
        now = timestamp or datetime.now(timezone.utc).isoformat()
        return cls(
            name=name,
            recipe=recipe,
            created_at=now,
            updated_at=now,
        )

    def with_recipe(
        self,
        recipe: ImageProcessingRecipe,
        *,
        timestamp: str | None = None,
    ) -> "ImageRecipePreset":
        return replace(
            self,
            recipe=recipe,
            updated_at=(
                timestamp or datetime.now(timezone.utc).isoformat()
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "recipe": self.recipe.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "ImageRecipePreset":
        if not isinstance(payload, dict):
            _raise(
                ImageRecipePresetErrorCode.INVALID_PRESET,
                "配方预设必须是 JSON 对象。",
            )
        unknown = sorted(
            set(payload) - {"name", "recipe", "created_at", "updated_at"}
        )
        if unknown:
            _raise(
                ImageRecipePresetErrorCode.INVALID_PRESET,
                f"配方预设包含未知字段：{'、'.join(unknown)}",
            )
        recipe_payload = payload.get("recipe")
        if not isinstance(recipe_payload, dict):
            _raise(
                ImageRecipePresetErrorCode.INVALID_PRESET,
                "配方预设缺少 recipe 对象。",
            )
        try:
            recipe = ImageProcessingRecipe.from_dict(recipe_payload)
            return cls(
                name=payload.get("name", ""),  # type: ignore[arg-type]
                recipe=recipe,
                created_at=payload.get("created_at", ""),  # type: ignore[arg-type]
                updated_at=payload.get("updated_at", ""),  # type: ignore[arg-type]
            )
        except ImageRecipePresetError:
            raise
        except (TypeError, ValueError) as exc:
            _raise(
                ImageRecipePresetErrorCode.INVALID_PRESET,
                f"配方预设内容无效：{exc}",
            )


def image_recipe_presets_path() -> Path:
    return settings_file_path().with_name(IMAGE_RECIPE_PRESET_FILENAME)


class ImageRecipePresetStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = (
            image_recipe_presets_path()
            if path is None
            else Path(path).expanduser()
        )

    def load(self) -> tuple[ImageRecipePreset, ...]:
        if not self.path.exists():
            return ()
        try:
            payload = json.loads(
                self.path.read_text(encoding="utf-8"),
                parse_constant=_reject_non_finite_constant,
            )
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            _raise(
                ImageRecipePresetErrorCode.INVALID_FILE,
                f"无法读取图像处理配方预设：{exc}",
            )
        if not isinstance(payload, dict):
            _raise(
                ImageRecipePresetErrorCode.INVALID_FILE,
                "图像处理配方文件必须是 JSON 对象。",
            )
        unknown = sorted(
            set(payload) - {"format", "schema_version", "presets"}
        )
        if unknown:
            _raise(
                ImageRecipePresetErrorCode.INVALID_FILE,
                f"配方文件包含未知字段：{'、'.join(unknown)}",
            )
        if payload.get("format") != IMAGE_RECIPE_PRESET_FORMAT:
            _raise(
                ImageRecipePresetErrorCode.INVALID_FILE,
                "文件不是 Fiber 图像处理配方预设。",
            )
        if payload.get("schema_version") != IMAGE_RECIPE_PRESET_SCHEMA_VERSION:
            _raise(
                ImageRecipePresetErrorCode.UNSUPPORTED_VERSION,
                "图像处理配方预设版本不受当前软件支持。",
            )
        raw_presets = payload.get("presets")
        if not isinstance(raw_presets, list):
            _raise(
                ImageRecipePresetErrorCode.INVALID_FILE,
                "配方文件的 presets 必须是列表。",
            )
        presets = tuple(
            ImageRecipePreset.from_dict(item)
            for item in raw_presets
        )
        _validate_unique_names(presets)
        return presets

    def save(
        self,
        presets: tuple[ImageRecipePreset, ...],
    ) -> Path:
        normalized = tuple(presets)
        if not all(isinstance(item, ImageRecipePreset) for item in normalized):
            raise TypeError("presets 必须全部是 ImageRecipePreset")
        _validate_unique_names(normalized)
        payload = {
            "format": IMAGE_RECIPE_PRESET_FORMAT,
            "schema_version": IMAGE_RECIPE_PRESET_SCHEMA_VERSION,
            "presets": [preset.to_dict() for preset in normalized],
        }
        try:
            return atomic_write_json(
                self.path,
                payload,
                ensure_ascii=False,
                indent=2,
            )
        except (OSError, TypeError, ValueError) as exc:
            _raise(
                ImageRecipePresetErrorCode.WRITE_FAILED,
                f"无法保存图像处理配方预设：{exc}",
            )

    def upsert(
        self,
        name: str,
        recipe: ImageProcessingRecipe,
        *,
        timestamp: str | None = None,
    ) -> ImageRecipePreset:
        normalized_name = _normalize_preset_name(name)
        presets = list(self.load())
        key = normalized_name.casefold()
        for index, preset in enumerate(presets):
            if preset.name.casefold() == key:
                updated = preset.with_recipe(
                    recipe,
                    timestamp=timestamp,
                )
                presets[index] = updated
                self.save(tuple(presets))
                return updated
        created = ImageRecipePreset.create(
            normalized_name,
            recipe,
            timestamp=timestamp,
        )
        presets.append(created)
        self.save(tuple(presets))
        return created

    def remove(self, name: str) -> bool:
        key = _normalize_preset_name(name).casefold()
        presets = list(self.load())
        retained = [
            preset
            for preset in presets
            if preset.name.casefold() != key
        ]
        if len(retained) == len(presets):
            return False
        self.save(tuple(retained))
        return True

    def get(self, name: str) -> ImageRecipePreset | None:
        key = _normalize_preset_name(name).casefold()
        return next(
            (
                preset
                for preset in self.load()
                if preset.name.casefold() == key
            ),
            None,
        )


def _validate_recipe(recipe: ImageProcessingRecipe) -> None:
    if not isinstance(recipe, ImageProcessingRecipe):
        raise TypeError("recipe 必须是 ImageProcessingRecipe")
    for operation in recipe.operations:
        try:
            resolved = ImageOperation(operation.operation_id)
        except ValueError:
            _raise(
                ImageRecipePresetErrorCode.UNKNOWN_OPERATION,
                f"配方包含当前版本未知操作：{operation.operation_id}",
            )
        if resolved is ImageOperation.IMAGE_CALCULATOR:
            _raise(
                ImageRecipePresetErrorCode.UNSUPPORTED_OPERATION,
                "批处理配方暂不支持图像计算器；"
                "它需要为每张图片单独指定第二幅对齐图像。",
            )


def _normalize_preset_name(value: object) -> str:
    name = unicodedata.normalize("NFC", str(value or "").strip())
    if not name:
        _raise(
            ImageRecipePresetErrorCode.INVALID_PRESET,
            "配方预设名称不能为空。",
        )
    if len(name) > 128 or any(character in "\r\n\t" for character in name):
        _raise(
            ImageRecipePresetErrorCode.INVALID_PRESET,
            "配方预设名称不能超过 128 个字符或包含控制字符。",
        )
    return name


def _normalize_timestamp(value: object, field_name: str) -> str:
    token = str(value or "").strip()
    if not token:
        _raise(
            ImageRecipePresetErrorCode.INVALID_PRESET,
            f"{field_name} 不能为空。",
        )
    try:
        parsed = datetime.fromisoformat(token.replace("Z", "+00:00"))
    except ValueError:
        _raise(
            ImageRecipePresetErrorCode.INVALID_PRESET,
            f"{field_name} 不是合法 ISO 时间。",
        )
    if parsed.tzinfo is None:
        _raise(
            ImageRecipePresetErrorCode.INVALID_PRESET,
            f"{field_name} 必须包含时区。",
        )
    return token


def _validate_unique_names(
    presets: tuple[ImageRecipePreset, ...],
) -> None:
    seen: set[str] = set()
    for preset in presets:
        key = preset.name.casefold()
        if key in seen:
            _raise(
                ImageRecipePresetErrorCode.DUPLICATE_NAME,
                f"配方预设名称重复：{preset.name}",
            )
        seen.add(key)


def _reject_non_finite_constant(value: str) -> None:
    raise ValueError(f"JSON 不允许非有限数：{value}")


def _raise(
    code: ImageRecipePresetErrorCode,
    message: str,
) -> None:
    raise ImageRecipePresetError(code, message)


__all__ = [
    "IMAGE_RECIPE_PRESET_FILENAME",
    "IMAGE_RECIPE_PRESET_FORMAT",
    "IMAGE_RECIPE_PRESET_SCHEMA_VERSION",
    "ImageRecipePreset",
    "ImageRecipePresetError",
    "ImageRecipePresetErrorCode",
    "ImageRecipePresetStore",
    "image_recipe_presets_path",
]
