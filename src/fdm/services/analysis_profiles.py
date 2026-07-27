"""Atomic named presets for Analyze measurement parameters.

Profiles deliberately live outside project files.  Loading or editing a
profile therefore cannot alter the numerical meaning of an existing project
or artifact.  A caller should still validate ``parameters`` against the
current tool parameter schema before starting a new analysis.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from types import MappingProxyType

from fdm.atomic_io import atomic_write_bytes
from fdm.settings import settings_file_path


ANALYSIS_PROFILE_STORE_SCHEMA_VERSION = 2
ANALYSIS_OUTPUT_FIELDS_PARAMETER = "__fdm_output_fields"
_PROFILE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_TOOL_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_OUTPUT_FIELD_KEY_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_MAX_STORE_BYTES = 4 << 20
_UNSET = object()


@dataclass(frozen=True, slots=True)
class AnalysisOutputFieldSpec:
    key: str
    chinese_name: str
    description: str = ""
    scalar_keys: tuple[str, ...] = ()
    table_names: tuple[str, ...] = ()
    asset_schemas: tuple[str, ...] = ()
    table_columns: tuple[tuple[str, str], ...] = ()
    table_row_labels: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        key = str(self.key or "").strip().lower()
        if not _OUTPUT_FIELD_KEY_PATTERN.fullmatch(key):
            raise ValueError(f"输出字段 key 格式无效：{self.key!r}")
        name = str(self.chinese_name or "").strip()
        if not name or len(name) > 128:
            raise ValueError("输出字段中文名称不能为空且不能超过 128 个字符")
        if any(ord(character) < 32 or ord(character) == 127 for character in name):
            raise ValueError("输出字段中文名称不能包含控制字符")
        description = str(self.description or "").strip()
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "chinese_name", name)
        object.__setattr__(self, "description", description)
        for attribute in (
            "scalar_keys",
            "table_names",
            "asset_schemas",
        ):
            values = tuple(str(value).strip() for value in getattr(self, attribute))
            if any(not value for value in values) or len(set(values)) != len(values):
                raise ValueError(f"{attribute} 不能包含空值或重复值")
            object.__setattr__(self, attribute, values)
        for attribute in ("table_columns", "table_row_labels"):
            values = tuple(
                (str(table).strip(), str(value).strip())
                for table, value in getattr(self, attribute)
            )
            if any(not table or not value for table, value in values):
                raise ValueError(f"{attribute} 不能包含空值")
            if len(set(values)) != len(values):
                raise ValueError(f"{attribute} 不能包含重复值")
            object.__setattr__(self, attribute, values)


@dataclass(frozen=True, slots=True)
class AnalysisOutputFieldSchema:
    tool_id: str
    fields: tuple[AnalysisOutputFieldSpec, ...]
    required_scalar_keys: tuple[str, ...] = ()
    required_table_columns: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def __post_init__(self) -> None:
        tool_id = str(self.tool_id or "").strip().lower()
        if not _TOOL_ID_PATTERN.fullmatch(tool_id):
            raise ValueError("输出字段 schema 的 tool_id 格式无效")
        fields = tuple(self.fields)
        if not fields or len({field.key for field in fields}) != len(fields):
            raise ValueError("输出字段 schema 必须包含唯一字段")
        required = tuple(str(value).strip() for value in self.required_scalar_keys)
        if any(not value for value in required) or len(set(required)) != len(required):
            raise ValueError("required_scalar_keys 不能包含空值或重复值")
        columns = tuple(
            (
                str(table).strip(),
                tuple(str(column).strip() for column in table_columns),
            )
            for table, table_columns in self.required_table_columns
        )
        if any(
            not table
            or not table_columns
            or any(not column for column in table_columns)
            for table, table_columns in columns
        ):
            raise ValueError("required_table_columns 格式无效")
        object.__setattr__(self, "tool_id", tool_id)
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "required_scalar_keys", required)
        object.__setattr__(self, "required_table_columns", columns)

    @property
    def default_fields(self) -> tuple[str, ...]:
        return tuple(field.key for field in self.fields)

    def normalize(
        self,
        fields: Iterable[str] | None,
        *,
        legacy_defaults: bool = True,
    ) -> tuple[str, ...] | None:
        if fields is None:
            return None if legacy_defaults else self.default_fields
        if isinstance(fields, (str, bytes)):
            raise TypeError("输出字段必须是字符串列表")
        requested = tuple(str(value).strip().lower() for value in fields)
        if len(set(requested)) != len(requested):
            raise ValueError("输出字段不能重复")
        known = {field.key for field in self.fields}
        unknown = set(requested) - known
        if unknown:
            raise ValueError(
                "包含未知输出字段：" + "、".join(sorted(unknown))
            )
        requested_set = set(requested)
        return tuple(
            field.key for field in self.fields if field.key in requested_set
        )


def _output_field(
    key: str,
    chinese_name: str,
    *,
    description: str = "",
    scalar_keys: tuple[str, ...] = (),
    table_names: tuple[str, ...] = (),
    asset_schemas: tuple[str, ...] = (),
    table_columns: tuple[tuple[str, str], ...] = (),
    table_row_labels: tuple[tuple[str, str], ...] = (),
) -> AnalysisOutputFieldSpec:
    return AnalysisOutputFieldSpec(
        key=key,
        chinese_name=chinese_name,
        description=description,
        scalar_keys=scalar_keys,
        table_names=table_names,
        asset_schemas=asset_schemas,
        table_columns=table_columns,
        table_row_labels=table_row_labels,
    )


ANALYSIS_OUTPUT_FIELD_SCHEMAS: Mapping[str, AnalysisOutputFieldSchema] = MappingProxyType({
    "fdm.shape": AnalysisOutputFieldSchema(
        tool_id="fdm.shape",
        required_scalar_keys=("unit", "area_from_exact_mask"),
        fields=(
            _output_field(
                "net_area",
                "净面积",
                scalar_keys=("area_px", "vector_area_px", "area"),
            ),
            _output_field(
                "hole_area",
                "孔洞面积",
                scalar_keys=("hole_area_px",),
            ),
            _output_field(
                "perimeter",
                "边界周长",
                scalar_keys=(
                    "outer_perimeter_px",
                    "hole_perimeter_px",
                    "total_perimeter_px",
                    "outer_perimeter",
                    "hole_perimeter",
                    "total_perimeter",
                ),
            ),
            _output_field(
                "topology",
                "组件、孔洞与 Euler 数",
                scalar_keys=("hole_count", "component_count", "euler_number"),
            ),
            _output_field(
                "equivalent_diameter",
                "等效圆直径",
                scalar_keys=("equivalent_circle_diameter",),
            ),
            _output_field(
                "feret",
                "Feret 直径与方向",
                scalar_keys=("feret_max", "feret_min", "feret_angle_degrees"),
            ),
            _output_field(
                "fitted_ellipse",
                "拟合椭圆",
                scalar_keys=(
                    "ellipse_major",
                    "ellipse_minor",
                    "ellipse_angle_degrees",
                ),
            ),
            _output_field(
                "shape_factors",
                "形状因子",
                scalar_keys=(
                    "extent",
                    "circularity",
                    "aspect_ratio",
                    "roundness",
                    "solidity",
                ),
            ),
            _output_field(
                "position_bounds",
                "质心与边界框",
                table_names=("位置与边界",),
            ),
            _output_field(
                "component_details",
                "分组件形状明细",
                table_names=("分组件形状指标",),
            ),
        ),
    ),
    "fdm.intensity": AnalysisOutputFieldSchema(
        tool_id="fdm.intensity",
        required_scalar_keys=(
            "included_pixel_count",
            "valid_pixel_count",
            "non_finite_count",
            "channel",
        ),
        fields=(
            _output_field(
                "central_tendency",
                "均值、中位数与众数",
                scalar_keys=("mean", "median", "mode"),
            ),
            _output_field(
                "dispersion_shape",
                "标准差、偏度与峰度",
                scalar_keys=("stddev", "skewness", "excess_kurtosis"),
            ),
            _output_field(
                "range",
                "最小值与最大值",
                scalar_keys=("minimum", "maximum"),
            ),
            _output_field(
                "integrated_density",
                "积分密度",
                scalar_keys=("integrated_density",),
            ),
            _output_field(
                "threshold_fraction",
                "阈值面积分数",
                scalar_keys=("threshold_area_fraction",),
            ),
            _output_field(
                "intensity_centroid",
                "强度重心",
                scalar_keys=(
                    "intensity_centroid_x_px",
                    "intensity_centroid_y_px",
                ),
            ),
            _output_field(
                "percentiles",
                "分位数",
                table_names=("分位数",),
            ),
            _output_field(
                "channel_statistics",
                "RGB 分通道统计",
                table_names=("通道统计",),
            ),
        ),
    ),
    "fdm.glcm": AnalysisOutputFieldSchema(
        tool_id="fdm.glcm",
        required_scalar_keys=(
            "levels",
            "quantization_minimum",
            "quantization_maximum",
            "symmetric",
            "valid_pixel_count",
            "non_finite_pixel_count",
        ),
        required_table_columns=(
            ("Haralick 特征", ("距离(px)", "方向(°)", "像素对数")),
        ),
        fields=tuple(
            _output_field(
                key,
                chinese_name,
                table_columns=(("Haralick 特征", column_name),),
                table_row_labels=(("Haralick 聚合", column_name),),
            )
            for key, chinese_name, column_name in (
                ("contrast", "Contrast 对比度", "Contrast"),
                ("dissimilarity", "Dissimilarity 差异性", "Dissimilarity"),
                ("homogeneity", "Homogeneity 同质性", "Homogeneity"),
                ("asm", "ASM 角二阶矩", "ASM"),
                ("energy", "Energy 能量", "Energy"),
                ("correlation", "Correlation 相关性", "Correlation"),
                ("entropy", "Entropy 熵", "Entropy"),
                (
                    "maximum_probability",
                    "Maximum Probability 最大概率",
                    "Maximum Probability",
                ),
            )
        )
        + (
            _output_field(
                "glcm_matrices",
                "原始 GLCM 矩阵资产",
                description="用于复核和后续分析；可能显著增加项目资产大小。",
                asset_schemas=("fdm.glcm-matrices.v1",),
            ),
        ),
    ),
})


def analysis_output_field_schema(
    tool_id: str,
) -> AnalysisOutputFieldSchema | None:
    return ANALYSIS_OUTPUT_FIELD_SCHEMAS.get(str(tool_id).strip().lower())


def normalize_analysis_output_fields(
    tool_id: str,
    fields: Iterable[str] | None,
    *,
    legacy_defaults: bool = True,
) -> tuple[str, ...] | None:
    schema = analysis_output_field_schema(tool_id)
    if schema is None:
        if fields is None:
            return None
        requested = tuple(fields)
        if requested:
            raise ValueError(f"{tool_id} 不支持输出字段选择")
        return ()
    return schema.normalize(fields, legacy_defaults=legacy_defaults)


@dataclass(frozen=True, slots=True, init=False)
class AnalysisMeasurementProfile:
    profile_id: str
    name: str
    tool_id: str
    tool_version: str
    created_at: str
    updated_at: str
    _parameters_json: str = field(repr=False)
    _output_fields_json: str | None = field(repr=False)

    def __init__(
        self,
        *,
        profile_id: str,
        name: str,
        tool_id: str,
        tool_version: str,
        parameters: Mapping[str, object],
        output_fields: Iterable[str] | None = None,
        created_at: str | None = None,
        updated_at: str | None = None,
    ) -> None:
        normalized_id = str(profile_id or "").strip()
        if not _PROFILE_ID_PATTERN.fullmatch(normalized_id):
            raise ValueError(
                "profile_id 必须以字母或数字开头，且仅包含字母、数字、点、"
                "下划线或连字符"
            )
        normalized_name = _required_text(name, "name", 128)
        normalized_tool = str(tool_id or "").strip().lower()
        if not _TOOL_ID_PATTERN.fullmatch(normalized_tool):
            raise ValueError("tool_id 格式无效")
        normalized_version = _required_text(tool_version, "tool_version", 64)
        if not isinstance(parameters, Mapping):
            raise TypeError("parameters 必须是对象")
        _validate_json_object_keys(parameters, "parameters")
        parameters_json = json.dumps(
            dict(parameters),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if len(parameters_json.encode("utf-8")) > 1 << 20:
            raise ValueError("profile parameters 不能超过 1 MiB")
        normalized_output_fields = normalize_analysis_output_fields(
            normalized_tool,
            output_fields,
            legacy_defaults=True,
        )
        output_fields_json = (
            None
            if normalized_output_fields is None
            else json.dumps(
                list(normalized_output_fields),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        )
        created = _timestamp(created_at or _now_iso(), "created_at")
        updated = _timestamp(updated_at or created, "updated_at")
        if datetime.fromisoformat(updated) < datetime.fromisoformat(created):
            raise ValueError("updated_at 不能早于 created_at")
        object.__setattr__(self, "profile_id", normalized_id)
        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "tool_id", normalized_tool)
        object.__setattr__(self, "tool_version", normalized_version)
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "updated_at", updated)
        object.__setattr__(self, "_parameters_json", parameters_json)
        object.__setattr__(self, "_output_fields_json", output_fields_json)

    @property
    def parameters(self) -> dict[str, object]:
        return json.loads(self._parameters_json)

    @property
    def output_fields(self) -> tuple[str, ...] | None:
        if self._output_fields_json is None:
            return None
        return tuple(json.loads(self._output_fields_json))

    def with_updates(
        self,
        *,
        name: str | None = None,
        tool_version: str | None = None,
        parameters: Mapping[str, object] | None = None,
        output_fields: Iterable[str] | None | object = _UNSET,
        updated_at: str | None = None,
    ) -> "AnalysisMeasurementProfile":
        return AnalysisMeasurementProfile(
            profile_id=self.profile_id,
            name=self.name if name is None else name,
            tool_id=self.tool_id,
            tool_version=(
                self.tool_version if tool_version is None else tool_version
            ),
            parameters=self.parameters if parameters is None else parameters,
            output_fields=(
                self.output_fields
                if output_fields is _UNSET
                else output_fields  # type: ignore[arg-type]
            ),
            created_at=self.created_at,
            updated_at=updated_at or _now_iso(),
        )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "profile_id": self.profile_id,
            "name": self.name,
            "tool_id": self.tool_id,
            "tool_version": self.tool_version,
            "parameters": self.parameters,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self.output_fields is not None:
            payload["output_fields"] = list(self.output_fields)
        return payload

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
    ) -> "AnalysisMeasurementProfile":
        required = {
                "profile_id",
                "name",
                "tool_id",
                "tool_version",
                "parameters",
                "created_at",
                "updated_at",
        }
        unknown = set(payload) - required - {"output_fields"}
        missing = required - set(payload)
        if missing:
            raise ValueError(f"profile 缺少字段: {', '.join(sorted(missing))}")
        if unknown:
            raise ValueError(f"profile 包含未知字段: {', '.join(sorted(unknown))}")
        parameters = payload["parameters"]
        if not isinstance(parameters, Mapping):
            raise TypeError("profile.parameters 必须是对象")
        return cls(
            profile_id=payload["profile_id"],  # type: ignore[arg-type]
            name=payload["name"],  # type: ignore[arg-type]
            tool_id=payload["tool_id"],  # type: ignore[arg-type]
            tool_version=payload["tool_version"],  # type: ignore[arg-type]
            parameters=parameters,
            output_fields=payload.get("output_fields"),  # type: ignore[arg-type]
            created_at=payload["created_at"],  # type: ignore[arg-type]
            updated_at=payload["updated_at"],  # type: ignore[arg-type]
        )


class AnalysisMeasurementProfileStore:
    """Read/write one bounded JSON document using atomic replacement."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = (
            analysis_measurement_profiles_path()
            if path is None
            else Path(path)
        )
        if not self.path.name:
            raise ValueError("profile store 路径无效")

    def load(self) -> tuple[AnalysisMeasurementProfile, ...]:
        if not self.path.exists():
            return ()
        if not self.path.is_file():
            raise ValueError("profile store 不是普通文件")
        if self.path.stat().st_size > _MAX_STORE_BYTES:
            raise ValueError("profile store 超过 4 MiB 安全上限")
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"无法读取分析预设：{exc}") from exc
        if not isinstance(payload, Mapping):
            raise TypeError("profile store 根节点必须是对象")
        _exact_keys(payload, {"schema_version", "profiles"}, "profile store")
        if payload["schema_version"] not in {
            1,
            ANALYSIS_PROFILE_STORE_SCHEMA_VERSION,
        }:
            raise ValueError(
                "不支持的 profile store schema_version: "
                f"{payload['schema_version']!r}"
            )
        profiles_payload = payload["profiles"]
        if not isinstance(profiles_payload, list) or any(
            not isinstance(item, Mapping) for item in profiles_payload
        ):
            raise TypeError("profile store.profiles 必须是对象列表")
        profiles = tuple(
            AnalysisMeasurementProfile.from_dict(item)
            for item in profiles_payload
        )
        _validate_profile_collection(profiles)
        return profiles

    def save(
        self,
        profiles: Iterable[AnalysisMeasurementProfile],
    ) -> tuple[AnalysisMeasurementProfile, ...]:
        frozen = tuple(profiles)
        _validate_profile_collection(frozen)
        payload = json.dumps(
            {
                "schema_version": ANALYSIS_PROFILE_STORE_SCHEMA_VERSION,
                "profiles": [profile.to_dict() for profile in frozen],
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(payload) > _MAX_STORE_BYTES:
            raise ValueError("profile store 超过 4 MiB 安全上限")
        atomic_write_bytes(self.path, payload)
        return frozen

    def upsert(
        self,
        profile: AnalysisMeasurementProfile,
    ) -> tuple[AnalysisMeasurementProfile, ...]:
        if not isinstance(profile, AnalysisMeasurementProfile):
            raise TypeError("profile 必须是 AnalysisMeasurementProfile")
        profiles = list(self.load())
        for index, existing in enumerate(profiles):
            if existing.profile_id == profile.profile_id:
                profiles[index] = profile
                break
        else:
            profiles.append(profile)
        profiles.sort(key=lambda item: (item.tool_id, item.name.casefold()))
        return self.save(profiles)

    def delete(self, profile_id: str) -> tuple[AnalysisMeasurementProfile, ...]:
        token = str(profile_id)
        profiles = tuple(
            profile
            for profile in self.load()
            if profile.profile_id != token
        )
        return self.save(profiles)


def _validate_profile_collection(
    profiles: tuple[AnalysisMeasurementProfile, ...],
) -> None:
    if any(not isinstance(item, AnalysisMeasurementProfile) for item in profiles):
        raise TypeError("profiles 必须全部是 AnalysisMeasurementProfile")
    ids = [item.profile_id for item in profiles]
    if len(set(ids)) != len(ids):
        raise ValueError("profile_id 不能重复")
    names = [
        (item.tool_id, item.tool_version, item.name.casefold())
        for item in profiles
    ]
    if len(set(names)) != len(names):
        raise ValueError("同一分析工具及版本下的预设名称不能重复")


def _validate_json_object_keys(value: object, field_name: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{field_name} 的对象键必须是字符串")
            _validate_json_object_keys(child, f"{field_name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_json_object_keys(child, f"{field_name}[{index}]")


def _required_text(value: object, name: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} 必须是字符串")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} 不能为空")
    if len(normalized) > maximum:
        raise ValueError(f"{name} 不能超过 {maximum} 个字符")
    if any(ord(character) < 32 or ord(character) == 127 for character in normalized):
        raise ValueError(f"{name} 不能包含控制字符")
    return normalized


def _timestamp(value: object, name: str) -> str:
    normalized = _required_text(value, name, 128)
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} 必须是 ISO 8601 时间") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} 必须包含时区")
    return parsed.isoformat()


def _exact_keys(
    payload: Mapping[str, object],
    required: set[str],
    name: str,
) -> None:
    missing = required - set(payload)
    unknown = set(payload) - required
    if missing:
        raise ValueError(f"{name} 缺少字段: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"{name} 包含未知字段: {', '.join(sorted(unknown))}")


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def analysis_measurement_profiles_path() -> Path:
    return settings_file_path().with_name("analysis-measurement-profiles.json")


__all__ = [
    "ANALYSIS_OUTPUT_FIELDS_PARAMETER",
    "ANALYSIS_OUTPUT_FIELD_SCHEMAS",
    "ANALYSIS_PROFILE_STORE_SCHEMA_VERSION",
    "AnalysisMeasurementProfile",
    "AnalysisMeasurementProfileStore",
    "AnalysisOutputFieldSchema",
    "AnalysisOutputFieldSpec",
    "analysis_output_field_schema",
    "analysis_measurement_profiles_path",
    "normalize_analysis_output_fields",
]
