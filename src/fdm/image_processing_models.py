from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import math
import re
from typing import Any, Callable

from fdm.raster import RasterPixelType


IMAGE_PROCESSING_SCHEMA_VERSION = 1
_OPERATION_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class RasterSemantic(str, Enum):
    """The meaning of samples carried through an image-processing recipe."""

    INTENSITY = "intensity"
    COLOR = "color"
    BINARY_MASK = "binary_mask"
    LABELS = "labels"
    DISTANCE = "distance"


@dataclass(frozen=True, slots=True)
class RasterTypeState:
    """Small, pixel-free state used to validate an operation type chain."""

    pixel_type: RasterPixelType
    semantic: RasterSemantic | None = None
    width: int | None = None
    height: int | None = None

    def __post_init__(self) -> None:
        pixel_type = RasterPixelType.parse(self.pixel_type)
        semantic = self.semantic
        if semantic is None:
            semantic = (
                RasterSemantic.INTENSITY
                if pixel_type.is_grayscale
                else RasterSemantic.COLOR
            )
        elif not isinstance(semantic, RasterSemantic):
            try:
                semantic = RasterSemantic(str(semantic))
            except ValueError as exc:
                raise ValueError(f"不支持的栅格语义: {self.semantic!r}") from exc
        if pixel_type.channel_count > 1 and semantic is not RasterSemantic.COLOR:
            raise ValueError("RGB/RGBA 栅格的语义必须是 color")
        if semantic is RasterSemantic.COLOR and pixel_type.is_grayscale:
            raise ValueError("color 语义必须使用 RGB8 或 RGBA8")
        width = _optional_positive_dimension(self.width, field_name="width")
        height = _optional_positive_dimension(self.height, field_name="height")
        if (width is None) != (height is None):
            raise ValueError("width 和 height 必须同时提供或同时省略")
        object.__setattr__(self, "pixel_type", pixel_type)
        object.__setattr__(self, "semantic", semantic)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)

    @property
    def channel_count(self) -> int:
        return self.pixel_type.channel_count

    @property
    def is_grayscale(self) -> bool:
        return self.pixel_type.is_grayscale

    def replace(
        self,
        *,
        pixel_type: RasterPixelType | str | None = None,
        semantic: RasterSemantic | str | None = None,
        width: int | None = None,
        height: int | None = None,
        preserve_dimensions: bool = True,
    ) -> "RasterTypeState":
        """Return a new state while retaining dimensions by default."""

        resolved_pixel_type = (
            self.pixel_type
            if pixel_type is None
            else RasterPixelType.parse(pixel_type)
        )
        resolved_semantic: RasterSemantic | None
        if semantic is None:
            resolved_semantic = self.semantic
        elif isinstance(semantic, RasterSemantic):
            resolved_semantic = semantic
        else:
            resolved_semantic = RasterSemantic(str(semantic))
        return RasterTypeState(
            pixel_type=resolved_pixel_type,
            semantic=resolved_semantic,
            width=self.width if preserve_dimensions and width is None else width,
            height=(
                self.height if preserve_dimensions and height is None else height
            ),
        )


class RoiProcessingSemantics(str, Enum):
    """How an operation treats an optional ROI mask."""

    UNSUPPORTED = "unsupported"
    WRITE_MASK_WITH_HALO = "write_mask_with_halo"
    ROI_STATISTICS = "roi_statistics"
    ISOLATED_DOMAIN = "isolated_domain"
    CROP_BOUNDS_OR_MASK = "crop_bounds_or_mask"

    # Source-compatible aliases for callers written before the scientific ROI
    # contract was made explicit. These are internal enum names, not persisted
    # recipe values.
    BLEND_WITH_SOURCE = WRITE_MASK_WITH_HALO
    BLEND_WITH_SCALAR_SOURCE = WRITE_MASK_WITH_HALO

    @property
    def supports_roi(self) -> bool:
        return self is not RoiProcessingSemantics.UNSUPPORTED


RasterInputCondition = Callable[
    [RasterTypeState, Mapping[str, object]],
    str | None,
]
RasterOutputResolver = Callable[
    [RasterTypeState, Mapping[str, object]],
    RasterTypeState,
]
TileCapabilityResolver = Callable[[Mapping[str, object]], object]
ImageOperationExecutor = Callable[..., object]


@dataclass(frozen=True, slots=True)
class ImageOperationParameterSchema:
    """Serializable, UI-independent contract for one operation parameter.

    Parameters remain optional by default so recipes written before the
    descriptor registry existed keep their executor-defined defaults.  New UI
    recipes materialise ``default`` values explicitly, while validation uses
    the same kind/choice/range contract in both paths.
    """

    key: str
    kind: str
    default: object = None
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[object, ...] = ()
    required: bool = False
    required_when: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        key = str(self.key or "").strip()
        if not key:
            raise ValueError("参数 schema 的 key 不能为空")
        kind = str(self.kind or "").strip().lower()
        if kind not in {
            "bool",
            "int",
            "float",
            "choice",
            "number_list",
            "secondary_image",
            "string",
        }:
            raise ValueError(f"参数 {key} 使用了未知类型：{kind}")
        minimum = self.minimum
        maximum = self.maximum
        for field_name, value in (("minimum", minimum), ("maximum", maximum)):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"参数 {key} 的 {field_name} 必须是有限数")
        if (
            minimum is not None
            and maximum is not None
            and float(minimum) > float(maximum)
        ):
            raise ValueError(f"参数 {key} 的 minimum 不能大于 maximum")
        choices = tuple(self.choices)
        if kind == "choice":
            if not choices:
                raise ValueError(f"选项参数 {key} 必须声明 choices")
            canonical_choices = {
                json.dumps(
                    _normalize_json_value(item, path=f"{key}.choices"),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                for item in choices
            }
            if len(canonical_choices) != len(choices):
                raise ValueError(f"选项参数 {key} 不能包含重复值")
        elif choices:
            raise ValueError(f"非选项参数 {key} 不能声明 choices")
        required_when = tuple(
            (str(other_key or "").strip(), expected)
            for other_key, expected in self.required_when
        )
        if any(not other_key for other_key, _expected in required_when):
            raise ValueError(f"参数 {key} 的 required_when 键不能为空")
        if len({other_key for other_key, _ in required_when}) != len(
            required_when
        ):
            raise ValueError(f"参数 {key} 的 required_when 不能重复")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)
        object.__setattr__(self, "choices", choices)
        object.__setattr__(self, "required_when", required_when)
        # Prove that persisted descriptor metadata is strict JSON even when a
        # default is a tuple used by a number-list editor.
        json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "key": self.key,
            "kind": self.kind,
            "default": _normalize_json_value(
                self.default,
                path=f"{self.key}.default",
            ),
            "required": bool(self.required),
        }
        if self.minimum is not None:
            payload["minimum"] = self.minimum
        if self.maximum is not None:
            payload["maximum"] = self.maximum
        if self.choices:
            payload["choices"] = [
                _normalize_json_value(item, path=f"{self.key}.choices")
                for item in self.choices
            ]
        if self.required_when:
            payload["required_when"] = {
                key: _normalize_json_value(
                    value,
                    path=f"{self.key}.required_when.{key}",
                )
                for key, value in self.required_when
            }
        return payload

    def validate_value(self, value: object) -> None:
        """Validate a provided JSON-compatible value without coercing it."""

        if self.kind == "bool":
            if not isinstance(value, bool):
                raise ValueError(f"参数 {self.key} 必须是布尔值")
            return
        if self.kind == "int":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"参数 {self.key} 必须是整数")
            numeric: int | float = value
        elif self.kind == "float":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"参数 {self.key} 必须是数值")
            try:
                numeric = float(value)
            except OverflowError as exc:
                raise ValueError(f"参数 {self.key} 必须是有限数") from exc
            if not math.isfinite(numeric):
                raise ValueError(f"参数 {self.key} 必须是有限数")
        elif self.kind == "choice":
            canonical_value = json.dumps(
                _normalize_json_value(value, path=self.key),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            canonical_choices = {
                json.dumps(
                    _normalize_json_value(choice, path=self.key),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                for choice in self.choices
            }
            if canonical_value not in canonical_choices:
                supported = "、".join(str(choice) for choice in self.choices)
                raise ValueError(
                    f"参数 {self.key} 的值不受支持：{value!r}；"
                    f"可选值为 {supported}"
                )
            return
        elif self.kind in {"secondary_image", "string"}:
            if not isinstance(value, str):
                raise ValueError(f"参数 {self.key} 必须是字符串")
            return
        elif self.kind == "number_list":
            if not isinstance(value, (list, tuple)) or not value:
                raise ValueError(f"参数 {self.key} 必须是非空数值列表")
            for item in value:
                if (
                    isinstance(item, bool)
                    or not isinstance(item, (int, float))
                    or not math.isfinite(float(item))
                ):
                    raise ValueError(f"参数 {self.key} 必须是有限数值列表")
            return
        else:  # pragma: no cover - guarded by __post_init__
            raise AssertionError(self.kind)
        if self.minimum is not None and numeric < float(self.minimum):
            raise ValueError(
                f"参数 {self.key} 不能小于 {self.minimum}"
            )
        if self.maximum is not None and numeric > float(self.maximum):
            raise ValueError(
                f"参数 {self.key} 不能大于 {self.maximum}"
            )


@dataclass(frozen=True, slots=True)
class ImageOperationDescriptor:
    """Complete non-UI contract for a registered image operation."""

    operation_id: str
    chinese_name: str
    category: str
    parameter_schema: tuple[ImageOperationParameterSchema, ...]
    input_conditions: tuple[RasterInputCondition, ...]
    output_resolver: RasterOutputResolver
    roi_semantics: RoiProcessingSemantics
    resource: str
    tile: TileCapabilityResolver
    executor: ImageOperationExecutor
    version: str = "1"

    def __post_init__(self) -> None:
        operation_id = str(self.operation_id or "").strip().lower()
        if not _OPERATION_ID_PATTERN.fullmatch(operation_id):
            raise ValueError(f"无效的图像操作 ID: {self.operation_id!r}")
        chinese_name = _required_text(
            self.chinese_name,
            field_name="chinese_name",
            maximum_length=128,
        )
        category = _required_text(
            self.category,
            field_name="category",
            maximum_length=128,
        )
        parameter_schema = tuple(self.parameter_schema)
        if not all(
            isinstance(item, ImageOperationParameterSchema)
            for item in parameter_schema
        ):
            raise TypeError(
                "parameter_schema 必须全部是 ImageOperationParameterSchema"
            )
        parameter_names = tuple(item.key for item in parameter_schema)
        if len(set(parameter_names)) != len(parameter_names):
            raise ValueError("parameter_schema 不能包含重复名称")
        conditions = tuple(self.input_conditions)
        if not all(callable(item) for item in conditions):
            raise TypeError("input_conditions 必须全部可调用")
        if not callable(self.output_resolver):
            raise TypeError("output_resolver 必须可调用")
        if not isinstance(self.roi_semantics, RoiProcessingSemantics):
            raise TypeError("roi_semantics 必须是 RoiProcessingSemantics")
        resource = _required_text(
            self.resource,
            field_name="resource",
            maximum_length=128,
        )
        if not callable(self.tile):
            raise TypeError("tile 必须可调用")
        if not callable(self.executor):
            raise TypeError("executor 必须可调用")
        version = _required_text(
            self.version,
            field_name="version",
            maximum_length=128,
        )
        object.__setattr__(self, "operation_id", operation_id)
        object.__setattr__(self, "chinese_name", chinese_name)
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "parameter_schema", parameter_schema)
        object.__setattr__(self, "input_conditions", conditions)
        object.__setattr__(self, "resource", resource)
        object.__setattr__(self, "version", version)

    @property
    def name(self) -> str:
        """Chinese operation name used by non-UI catalog consumers."""

        return self.chinese_name

    @property
    def parameters(self) -> tuple[str, ...]:
        """Backward-compatible ordered parameter-name view."""

        return tuple(item.key for item in self.parameter_schema)

    def parameter(self, key: str) -> ImageOperationParameterSchema:
        resolved = str(key)
        for item in self.parameter_schema:
            if item.key == resolved:
                return item
        raise KeyError(resolved)

    def validate_parameters(
        self,
        parameters: Mapping[str, object],
    ) -> None:
        if not isinstance(parameters, Mapping):
            raise TypeError("parameters 必须是对象")
        schemas = {item.key: item for item in self.parameter_schema}
        unknown = sorted(set(parameters) - set(schemas))
        if unknown:
            raise ValueError("包含未声明参数：" + "、".join(unknown))
        resolved = {
            item.key: parameters.get(item.key, item.default)
            for item in self.parameter_schema
        }
        missing = []
        for item in self.parameter_schema:
            condition_matches = bool(item.required_when) and all(
                resolved.get(other_key) == expected
                for other_key, expected in item.required_when
            )
            if (
                (item.required or condition_matches)
                and (
                    item.key not in parameters
                    or parameters.get(item.key) is None
                    or parameters.get(item.key) == ""
                )
            ):
                missing.append(item.key)
        if missing:
            raise ValueError("缺少必填参数：" + "、".join(missing))
        for key, value in parameters.items():
            schemas[key].validate_value(value)

    def validate_input(
        self,
        state: RasterTypeState,
        parameters: Mapping[str, object],
    ) -> None:
        if not isinstance(state, RasterTypeState):
            raise TypeError("state 必须是 RasterTypeState")
        if not isinstance(parameters, Mapping):
            raise TypeError("parameters 必须是对象")
        self.validate_parameters(parameters)
        for condition in self.input_conditions:
            error = condition(state, parameters)
            if error:
                raise ValueError(str(error))

    def resolve_output(
        self,
        state: RasterTypeState,
        parameters: Mapping[str, object],
    ) -> RasterTypeState:
        self.validate_input(state, parameters)
        output = self.output_resolver(state, parameters)
        if not isinstance(output, RasterTypeState):
            raise TypeError(
                f"操作 {self.operation_id} 的 output_resolver "
                "必须返回 RasterTypeState"
            )
        return output


@dataclass(frozen=True, slots=True)
class DisplayTransform:
    """A non-destructive mapping from stored samples to screen intensity.

    The transform describes presentation only.  It must never be used as a
    substitute for a committed image operation, nor may it change measurement
    geometry or calibrated values.
    """

    black_point: float | None = None
    white_point: float | None = None
    channel_ranges: tuple[tuple[float, float], ...] = ()
    gamma: float = 1.0
    lut_id: str | None = None
    window_center: float | None = None
    window_width: float | None = None
    inverted: bool = False
    schema_version: int = field(
        default=IMAGE_PROCESSING_SCHEMA_VERSION,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _require_supported_schema(self.schema_version, type_name="DisplayTransform")
        black_point = _optional_finite(self.black_point, field_name="black_point")
        white_point = _optional_finite(self.white_point, field_name="white_point")
        if (black_point is None) != (white_point is None):
            raise ValueError("black_point 和 white_point 必须同时提供")
        if (
            black_point is not None
            and white_point is not None
            and black_point >= white_point
        ):
            raise ValueError("white_point 必须大于 black_point")
        channel_ranges = _normalize_display_channel_ranges(self.channel_ranges)
        if black_point is not None and channel_ranges:
            raise ValueError(
                "旧版全局显示范围与各通道显示范围不能同时提供"
            )
        window_center = _optional_finite(
            self.window_center,
            field_name="window_center",
        )
        window_width = _optional_finite(
            self.window_width,
            field_name="window_width",
        )
        if (window_center is None) != (window_width is None):
            raise ValueError("window_center 和 window_width 必须同时提供")
        if window_width is not None and window_width <= 0.0:
            raise ValueError("window_width 必须是正有限数值")
        if window_center is not None and (
            black_point is not None or channel_ranges
        ):
            raise ValueError("窗宽/窗位与显式显示范围不能同时提供")
        gamma = _positive_finite(self.gamma, field_name="gamma")
        lut_id = _normalize_display_lut_id(self.lut_id)
        if not isinstance(self.inverted, bool):
            raise TypeError("inverted 必须是布尔值")
        object.__setattr__(self, "black_point", black_point)
        object.__setattr__(self, "white_point", white_point)
        object.__setattr__(self, "channel_ranges", channel_ranges)
        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(self, "lut_id", lut_id)
        object.__setattr__(self, "window_center", window_center)
        object.__setattr__(self, "window_width", window_width)
        object.__setattr__(self, "schema_version", IMAGE_PROCESSING_SCHEMA_VERSION)

    @property
    def effective_channel_ranges(self) -> tuple[tuple[float, float], ...]:
        """Return explicit ranges without guessing a raster's native domain."""

        if self.channel_ranges:
            return self.channel_ranges
        if self.black_point is not None and self.white_point is not None:
            return ((self.black_point, self.white_point),)
        if self.window_center is not None and self.window_width is not None:
            half_width = self.window_width / 2.0
            return (
                (
                    self.window_center - half_width,
                    self.window_center + half_width,
                ),
            )
        return ()

    def ranges_for_pixel_type(
        self,
        pixel_type: RasterPixelType | str,
    ) -> tuple[tuple[float, float], ...]:
        """Validate and expand presentation ranges for one raster layout.

        A one-channel legacy range remains valid for RGB/RGBA and is broadcast
        to the three colour channels.  Alpha is deliberately excluded.
        """

        parsed = RasterPixelType.parse(pixel_type)
        ranges = self.effective_channel_ranges
        if parsed.is_grayscale:
            if len(ranges) > 1:
                raise ValueError("灰度图片只能使用一个通道显示范围")
            return ranges
        if self.window_center is not None:
            raise ValueError("窗宽/窗位只适用于灰度图片")
        if self.lut_id not in {None, "grayscale"}:
            raise ValueError("彩色图片不能应用灰度 LUT")
        if not ranges:
            return ()
        if len(ranges) == 1:
            return ranges * 3
        if len(ranges) != 3:
            raise ValueError("RGB/RGBA 图片必须提供一个或三个通道显示范围")
        return ranges

    @property
    def is_identity(self) -> bool:
        return (
            self.black_point is None
            and self.white_point is None
            and not self.channel_ranges
            and self.gamma == 1.0
            and self.lut_id is None
            and self.window_center is None
            and self.window_width is None
            and not self.inverted
        )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": IMAGE_PROCESSING_SCHEMA_VERSION,
            "gamma": self.gamma,
            "inverted": self.inverted,
        }
        if self.black_point is not None and self.white_point is not None:
            payload["black_point"] = self.black_point
            payload["white_point"] = self.white_point
        if self.channel_ranges:
            payload["channel_ranges"] = [
                [low, high]
                for low, high in self.channel_ranges
            ]
        if self.lut_id is not None:
            payload["lut_id"] = self.lut_id
        if self.window_center is not None and self.window_width is not None:
            payload["window_center"] = self.window_center
            payload["window_width"] = self.window_width
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "DisplayTransform":
        if not isinstance(payload, Mapping):
            raise TypeError("DisplayTransform payload 必须是对象")
        return cls(
            black_point=payload.get("black_point"),
            white_point=payload.get("white_point"),
            channel_ranges=payload.get("channel_ranges", ()),  # type: ignore[arg-type]
            gamma=payload.get("gamma", 1.0),
            lut_id=payload.get("lut_id"),  # type: ignore[arg-type]
            window_center=payload.get("window_center"),
            window_width=payload.get("window_width"),
            inverted=payload.get("inverted", False),
            schema_version=payload.get(
                "schema_version",
                IMAGE_PROCESSING_SCHEMA_VERSION,
            ),
        )


_DISPLAY_LUT_ALIASES = {
    "gray": "grayscale",
    "grey": "grayscale",
    "grayscale": "grayscale",
    "red": "red",
    "green": "green",
    "blue": "blue",
    "fire": "fire",
    "ice": "ice",
    "spectrum": "spectrum",
}


def _normalize_display_lut_id(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("lut_id 必须是字符串")
    token = value.strip().casefold()
    if not token:
        return None
    try:
        return _DISPLAY_LUT_ALIASES[token]
    except KeyError as exc:
        supported = "、".join(sorted(set(_DISPLAY_LUT_ALIASES.values())))
        raise ValueError(f"不支持的显示 LUT：{value}；可选 {supported}") from exc


def _normalize_display_channel_ranges(
    value: Iterable[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    if value is None:  # type: ignore[comparison-overlap]
        return ()
    try:
        items = tuple(value)
    except TypeError as exc:
        raise TypeError("channel_ranges 必须是通道范围序列") from exc
    if len(items) not in {0, 1, 3}:
        raise ValueError("channel_ranges 只能包含一个或三个通道范围")
    normalized: list[tuple[float, float]] = []
    for index, item in enumerate(items):
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise TypeError(
                f"channel_ranges[{index}] 必须是 [最小值, 最大值]"
            )
        low = _optional_finite(
            item[0],
            field_name=f"channel_ranges[{index}].minimum",
        )
        high = _optional_finite(
            item[1],
            field_name=f"channel_ranges[{index}].maximum",
        )
        if low is None or high is None:  # pragma: no cover - item values exist
            raise TypeError("通道显示范围不能为 null")
        if high <= low:
            raise ValueError(
                f"channel_ranges[{index}] 的最大值必须大于最小值"
            )
        normalized.append((low, high))
    return tuple(normalized)


@dataclass(frozen=True, slots=True, init=False)
class ImageOperationSpec:
    """One deterministic, versioned image-processing operation.

    Parameters are stored internally as canonical JSON.  Callers receive a new
    dictionary from :attr:`parameters`, so neither a caller-owned mapping nor a
    nested list can mutate an in-flight recipe after it is handed to a worker.
    """

    operation_id: str
    implementation: str
    implementation_version: str
    _parameters_json: str = field(repr=False)
    _result_metadata_json: str = field(repr=False)

    def __init__(
        self,
        operation_id: str,
        parameters: Mapping[str, object] | None = None,
        *,
        implementation: str = "fdm",
        implementation_version: str = "1",
        result_metadata: Mapping[str, object] | None = None,
    ) -> None:
        normalized_operation_id = str(operation_id or "").strip().lower()
        if not _OPERATION_ID_PATTERN.fullmatch(normalized_operation_id):
            raise ValueError(
                "operation_id 必须是小写字母或数字开头，且仅包含 "
                "a-z、0-9、点、下划线或连字符"
            )
        normalized_implementation = _required_text(
            implementation,
            field_name="implementation",
            maximum_length=128,
        )
        normalized_version = _required_text(
            implementation_version,
            field_name="implementation_version",
            maximum_length=128,
        )
        normalized_parameters = _normalize_json_object(
            parameters if parameters is not None else {},
            field_name="parameters",
        )
        normalized_result_metadata = _normalize_json_object(
            result_metadata if result_metadata is not None else {},
            field_name="result_metadata",
        )
        parameters_json = json.dumps(
            normalized_parameters,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        result_metadata_json = json.dumps(
            normalized_result_metadata,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        object.__setattr__(self, "operation_id", normalized_operation_id)
        object.__setattr__(self, "implementation", normalized_implementation)
        object.__setattr__(self, "implementation_version", normalized_version)
        object.__setattr__(self, "_parameters_json", parameters_json)
        object.__setattr__(
            self,
            "_result_metadata_json",
            result_metadata_json,
        )

    @property
    def parameters(self) -> dict[str, object]:
        return json.loads(self._parameters_json)

    @property
    def result_metadata(self) -> dict[str, object]:
        """Return immutable-at-rest audit facts produced by this operation."""

        return json.loads(self._result_metadata_json)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "operation_id": self.operation_id,
            "parameters": self.parameters,
            "implementation": self.implementation,
            "implementation_version": self.implementation_version,
        }
        if self.result_metadata:
            payload["result_metadata"] = self.result_metadata
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ImageOperationSpec":
        if not isinstance(payload, Mapping):
            raise TypeError("ImageOperationSpec payload 必须是对象")
        parameters = payload.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise TypeError("ImageOperationSpec.parameters 必须是对象")
        result_metadata = payload.get("result_metadata", {})
        if not isinstance(result_metadata, Mapping):
            raise TypeError("ImageOperationSpec.result_metadata 必须是对象")
        return cls(
            operation_id=payload.get("operation_id", ""),  # type: ignore[arg-type]
            parameters=parameters,
            implementation=payload.get(  # type: ignore[arg-type]
                "implementation",
                "fdm",
            ),
            implementation_version=payload.get(  # type: ignore[arg-type]
                "implementation_version",
                "1",
            ),
            result_metadata=result_metadata,
        )


@dataclass(frozen=True, slots=True)
class ImageProcessingRecipe:
    """An ordered, immutable list of operations applied to one source raster."""

    operations: tuple[ImageOperationSpec, ...]
    schema_version: int = field(
        default=IMAGE_PROCESSING_SCHEMA_VERSION,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _require_supported_schema(
            self.schema_version,
            type_name="ImageProcessingRecipe",
        )
        operations = tuple(self.operations)
        if not operations:
            raise ValueError("图像处理配方至少需要一个操作")
        if not all(isinstance(item, ImageOperationSpec) for item in operations):
            raise TypeError("operations 必须全部是 ImageOperationSpec")
        object.__setattr__(self, "operations", operations)
        object.__setattr__(self, "schema_version", IMAGE_PROCESSING_SCHEMA_VERSION)

    @classmethod
    def from_operations(
        cls,
        operations: Iterable[ImageOperationSpec],
    ) -> "ImageProcessingRecipe":
        return cls(operations=tuple(operations))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": IMAGE_PROCESSING_SCHEMA_VERSION,
            "operations": [operation.to_dict() for operation in self.operations],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ImageProcessingRecipe":
        if not isinstance(payload, Mapping):
            raise TypeError("ImageProcessingRecipe payload 必须是对象")
        schema_version = payload.get(
            "schema_version",
            IMAGE_PROCESSING_SCHEMA_VERSION,
        )
        _require_supported_schema(
            schema_version,
            type_name="ImageProcessingRecipe",
        )
        operations_payload = payload.get("operations")
        if not isinstance(operations_payload, list):
            raise TypeError("ImageProcessingRecipe.operations 必须是列表")
        if not all(isinstance(item, Mapping) for item in operations_payload):
            raise TypeError(
                "ImageProcessingRecipe.operations 必须全部是对象"
            )
        return cls(
            operations=tuple(
                ImageOperationSpec.from_dict(item)
                for item in operations_payload
            ),
            schema_version=IMAGE_PROCESSING_SCHEMA_VERSION,
        )


@dataclass(frozen=True, slots=True)
class ProcessingRoiSnapshot:
    """Exact ROI dependency frozen when a processing session starts."""

    source_kind: str
    source_id: str
    revision: int
    bounds: tuple[int, int, int, int]
    mask_sha256: str
    dependency_revisions: tuple[tuple[str, int], ...] = ()
    source_label: str = ""

    def __post_init__(self) -> None:
        kind = _required_text(
            self.source_kind,
            field_name="roi_snapshot.source_kind",
            maximum_length=64,
        )
        if kind not in {"project_roi", "measurement_area"}:
            raise ValueError("roi_snapshot.source_kind 不受支持")
        source_id = _required_text(
            self.source_id,
            field_name="roi_snapshot.source_id",
            maximum_length=256,
        )
        revision = _nonnegative_int(
            self.revision,
            field_name="roi_snapshot.revision",
        )
        if len(self.bounds) != 4:
            raise ValueError("roi_snapshot.bounds 必须包含 x、y、width、height")
        bounds = tuple(int(value) for value in self.bounds)
        if (
            bounds[0] < 0
            or bounds[1] < 0
            or bounds[2] <= 0
            or bounds[3] <= 0
        ):
            raise ValueError("roi_snapshot.bounds 必须是非空的非负像素范围")
        digest = str(self.mask_sha256 or "").strip().lower()
        if not _SHA256_PATTERN.fullmatch(digest):
            raise ValueError("roi_snapshot.mask_sha256 必须是 SHA256")
        dependencies = tuple(
            (
                _required_text(
                    item_id,
                    field_name="roi_snapshot.dependency_id",
                    maximum_length=256,
                ),
                _nonnegative_int(
                    item_revision,
                    field_name="roi_snapshot.dependency_revision",
                ),
            )
            for item_id, item_revision in self.dependency_revisions
        )
        if len({item_id for item_id, _revision in dependencies}) != len(
            dependencies
        ):
            raise ValueError("roi_snapshot.dependency_revisions 不能重复")
        object.__setattr__(self, "source_kind", kind)
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(self, "mask_sha256", digest)
        object.__setattr__(
            self,
            "dependency_revisions",
            tuple(sorted(dependencies)),
        )
        object.__setattr__(
            self,
            "source_label",
            str(self.source_label or "").strip(),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "revision": self.revision,
            "bounds": list(self.bounds),
            "mask_sha256": self.mask_sha256,
            "dependency_revisions": {
                item_id: revision
                for item_id, revision in self.dependency_revisions
            },
            "source_label": self.source_label,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
    ) -> "ProcessingRoiSnapshot":
        dependencies = payload.get("dependency_revisions", {})
        if not isinstance(dependencies, Mapping):
            raise TypeError("roi_snapshot.dependency_revisions 必须是对象")
        bounds = payload.get("bounds")
        if not isinstance(bounds, (list, tuple)):
            raise TypeError("roi_snapshot.bounds 必须是列表")
        return cls(
            source_kind=payload.get("source_kind", ""),  # type: ignore[arg-type]
            source_id=payload.get("source_id", ""),  # type: ignore[arg-type]
            revision=payload.get("revision", 0),  # type: ignore[arg-type]
            bounds=tuple(bounds),  # type: ignore[arg-type]
            mask_sha256=payload.get("mask_sha256", ""),  # type: ignore[arg-type]
            dependency_revisions=tuple(
                (str(item_id), int(revision))
                for item_id, revision in dependencies.items()
            ),
            source_label=payload.get("source_label", ""),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class ImageDerivation:
    """Audit metadata for an authoritative, persisted derived raster.

    The recipe documents how the result was produced, while the stored project
    asset remains authoritative.  Reopening a project therefore never depends
    on replaying an operation with a potentially different library version.
    """

    source_document_id: str
    recipe: ImageProcessingRecipe
    source_path: str | None = None
    source_sha256: str | None = None
    source_image_size: tuple[int, int] | None = None
    source_pixel_revision: int = 0
    source_pixel_type: RasterPixelType | None = None
    result_pixel_type: RasterPixelType | None = None
    result_image_size: tuple[int, int] | None = None
    result_sha256: str | None = None
    roi_snapshot: ProcessingRoiSnapshot | None = None
    library_versions: tuple[tuple[str, str], ...] = ()
    created_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    schema_version: int = field(
        default=IMAGE_PROCESSING_SCHEMA_VERSION,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _require_supported_schema(
            self.schema_version,
            type_name="ImageDerivation",
        )
        source_document_id = _required_text(
            self.source_document_id,
            field_name="source_document_id",
            maximum_length=256,
        )
        if not isinstance(self.recipe, ImageProcessingRecipe):
            raise TypeError("recipe 必须是 ImageProcessingRecipe")
        source_path = _optional_text(self.source_path, maximum_length=32_768)
        source_sha256 = _optional_sha256(
            self.source_sha256,
            field_name="source_sha256",
        )
        result_sha256 = _optional_sha256(
            self.result_sha256,
            field_name="result_sha256",
        )
        if self.roi_snapshot is not None and not isinstance(
            self.roi_snapshot,
            ProcessingRoiSnapshot,
        ):
            raise TypeError("roi_snapshot 必须是 ProcessingRoiSnapshot")
        source_image_size = _optional_image_size(self.source_image_size)
        source_pixel_revision = _nonnegative_int(
            self.source_pixel_revision,
            field_name="source_pixel_revision",
        )
        source_pixel_type = _optional_pixel_type(self.source_pixel_type)
        result_pixel_type = _optional_pixel_type(self.result_pixel_type)
        result_image_size = _optional_image_size(
            self.result_image_size,
            field_name="result_image_size",
        )
        library_versions = _normalize_version_pairs(self.library_versions)
        created_at = _required_text(
            self.created_at,
            field_name="created_at",
            maximum_length=128,
        )
        object.__setattr__(self, "source_document_id", source_document_id)
        object.__setattr__(self, "source_path", source_path)
        object.__setattr__(self, "source_sha256", source_sha256)
        object.__setattr__(self, "source_image_size", source_image_size)
        object.__setattr__(self, "source_pixel_revision", source_pixel_revision)
        object.__setattr__(self, "source_pixel_type", source_pixel_type)
        object.__setattr__(self, "result_pixel_type", result_pixel_type)
        object.__setattr__(self, "result_image_size", result_image_size)
        object.__setattr__(self, "result_sha256", result_sha256)
        object.__setattr__(self, "library_versions", library_versions)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "schema_version", IMAGE_PROCESSING_SCHEMA_VERSION)

    def to_dict(self) -> dict[str, object]:
        source: dict[str, object] = {
            "document_id": self.source_document_id,
        }
        if self.source_path is not None:
            source["path"] = self.source_path
        if self.source_sha256 is not None:
            source["sha256"] = self.source_sha256
        if self.source_image_size is not None:
            source["image_size"] = list(self.source_image_size)
        source["pixel_revision"] = self.source_pixel_revision
        if self.source_pixel_type is not None:
            source["pixel_type"] = self.source_pixel_type.value

        result: dict[str, object] = {}
        if self.result_pixel_type is not None:
            result["pixel_type"] = self.result_pixel_type.value
        if self.result_image_size is not None:
            result["image_size"] = list(self.result_image_size)
        if self.result_sha256 is not None:
            result["sha256"] = self.result_sha256

        payload: dict[str, object] = {
            "schema_version": IMAGE_PROCESSING_SCHEMA_VERSION,
            "source": source,
            "recipe": self.recipe.to_dict(),
            "created_at": self.created_at,
            "library_versions": dict(self.library_versions),
        }
        if result:
            payload["result"] = result
        if self.roi_snapshot is not None:
            payload["roi_snapshot"] = self.roi_snapshot.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ImageDerivation":
        if not isinstance(payload, Mapping):
            raise TypeError("ImageDerivation payload 必须是对象")
        _require_supported_schema(
            payload.get("schema_version", IMAGE_PROCESSING_SCHEMA_VERSION),
            type_name="ImageDerivation",
        )
        source = payload.get("source")
        if not isinstance(source, Mapping):
            raise TypeError("ImageDerivation.source 必须是对象")
        recipe_payload = payload.get("recipe")
        if not isinstance(recipe_payload, Mapping):
            raise TypeError("ImageDerivation.recipe 必须是对象")
        result = payload.get("result", {})
        if not isinstance(result, Mapping):
            raise TypeError("ImageDerivation.result 必须是对象")
        source_size_payload = source.get("image_size")
        source_size = (
            tuple(source_size_payload)
            if isinstance(source_size_payload, (list, tuple))
            else None
        )
        result_size_payload = result.get("image_size")
        result_size = (
            tuple(result_size_payload)
            if isinstance(result_size_payload, (list, tuple))
            else None
        )
        versions_payload = payload.get("library_versions", {})
        if not isinstance(versions_payload, Mapping):
            raise TypeError("ImageDerivation.library_versions 必须是对象")
        roi_snapshot_payload = payload.get("roi_snapshot")
        if (
            roi_snapshot_payload is not None
            and not isinstance(roi_snapshot_payload, Mapping)
        ):
            raise TypeError("ImageDerivation.roi_snapshot 必须是对象")
        return cls(
            source_document_id=source.get("document_id", ""),  # type: ignore[arg-type]
            source_path=(
                str(source["path"])
                if source.get("path") is not None
                else None
            ),
            source_sha256=(
                str(source["sha256"])
                if source.get("sha256") is not None
                else None
            ),
            source_image_size=source_size,
            source_pixel_revision=source.get("pixel_revision", 0),  # type: ignore[arg-type]
            source_pixel_type=source.get("pixel_type"),
            recipe=ImageProcessingRecipe.from_dict(recipe_payload),
            result_pixel_type=result.get("pixel_type"),
            result_image_size=result_size,
            result_sha256=(
                str(result["sha256"])
                if result.get("sha256") is not None
                else None
            ),
            roi_snapshot=(
                None
                if roi_snapshot_payload is None
                else ProcessingRoiSnapshot.from_dict(roi_snapshot_payload)
            ),
            library_versions=tuple(
                (str(key), str(value))
                for key, value in versions_payload.items()
            ),
            created_at=payload.get("created_at", ""),  # type: ignore[arg-type]
            schema_version=IMAGE_PROCESSING_SCHEMA_VERSION,
        )


def _require_supported_schema(value: object, *, type_name: str) -> None:
    if isinstance(value, bool):
        raise ValueError(f"{type_name}.schema_version 无效")
    try:
        version = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{type_name}.schema_version 无效") from exc
    if version != IMAGE_PROCESSING_SCHEMA_VERSION:
        raise ValueError(f"不支持的 {type_name} schema_version: {version}")


def _optional_finite(value: object, *, field_name: str) -> float | None:
    if value is None:
        return None
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} 必须是有限数值") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} 必须是有限数值")
    return normalized


def _positive_finite(value: object, *, field_name: str) -> float:
    normalized = _optional_finite(value, field_name=field_name)
    if normalized is None or normalized <= 0.0:
        raise ValueError(f"{field_name} 必须是大于 0 的有限数值")
    return normalized


def _required_text(
    value: object,
    *,
    field_name: str,
    maximum_length: int,
) -> str:
    token = str(value or "").strip()
    if not token:
        raise ValueError(f"{field_name} 不能为空")
    if len(token) > maximum_length:
        raise ValueError(f"{field_name} 超过 {maximum_length} 字符上限")
    return token


def _optional_text(value: object, *, maximum_length: int) -> str | None:
    if value is None:
        return None
    token = str(value).strip()
    if not token:
        return None
    if len(token) > maximum_length:
        raise ValueError(f"文本超过 {maximum_length} 字符上限")
    return token


def _optional_sha256(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    token = str(value).strip().lower()
    if not _SHA256_PATTERN.fullmatch(token):
        raise ValueError(f"{field_name} 必须是 64 位 SHA256 十六进制字符串")
    return token


def _optional_image_size(
    value: object,
    *,
    field_name: str = "source_image_size",
) -> tuple[int, int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field_name} 必须包含宽度和高度")
    width = _positive_dimension(value[0], field_name=f"{field_name}.width")
    height = _positive_dimension(value[1], field_name=f"{field_name}.height")
    return (width, height)


def _positive_dimension(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} 必须是正整数")
    try:
        normalized = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} 必须是正整数") from exc
    if normalized != value or normalized <= 0:
        raise ValueError(f"{field_name} 必须是正整数")
    return normalized


def _optional_positive_dimension(
    value: object,
    *,
    field_name: str,
) -> int | None:
    if value is None:
        return None
    return _positive_dimension(value, field_name=field_name)


def _nonnegative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} 必须是非负整数")
    try:
        normalized = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} 必须是非负整数") from exc
    if normalized != value or normalized < 0:
        raise ValueError(f"{field_name} 必须是非负整数")
    return normalized


def _normalize_version_pairs(
    value: Iterable[tuple[object, object]],
) -> tuple[tuple[str, str], ...]:
    try:
        pairs = tuple(value)
    except TypeError as exc:
        raise TypeError("library_versions 必须是键值对序列") from exc
    normalized: dict[str, str] = {}
    for item in pairs:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise TypeError("library_versions 必须是键值对序列")
        key = _required_text(
            item[0],
            field_name="library_versions.name",
            maximum_length=128,
        )
        version = _required_text(
            item[1],
            field_name=f"library_versions.{key}",
            maximum_length=128,
        )
        normalized[key] = version
    return tuple(sorted(normalized.items()))


def _optional_pixel_type(value: object) -> RasterPixelType | None:
    if value is None:
        return None
    return RasterPixelType.parse(value)


def _normalize_json_object(
    value: Mapping[str, object],
    *,
    field_name: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} 必须是对象")
    normalized = _normalize_json_value(value, path=field_name)
    if not isinstance(normalized, dict):  # pragma: no cover - guarded above
        raise TypeError(f"{field_name} 必须是对象")
    return normalized


def _normalize_json_value(value: object, *, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} 不允许 NaN 或 Inf")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} 的键必须是字符串")
            normalized[key] = _normalize_json_value(
                item,
                path=f"{path}.{key}",
            )
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _normalize_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{path} 包含不可序列化的值: {type(value).__name__}")
