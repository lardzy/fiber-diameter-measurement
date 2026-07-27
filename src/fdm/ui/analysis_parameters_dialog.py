"""Schema-driven Chinese parameter editor for all Analyze tools."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
import json
import math
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPalette, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from fdm.analysis_artifacts import AnalysisCurve
from fdm.cancellation import CancellationToken
from fdm.raster import RasterPixelType, RasterPlane
from fdm.services.analysis_profiles import AnalysisMeasurementProfileStore
from fdm.ui.analysis_profile_controls import (
    AnalysisOutputFieldSelector,
    AnalysisProfileControls,
)
from fdm.ui.image_analysis_controller import (
    AnalysisCalibrationSnapshot,
    AnalysisPhaseCallback,
    AnalysisTool,
    ImageAnalysisTaskController,
    ImageAnalysisTaskRequest,
    ImageAnalysisTaskResult,
    execute_analysis_task,
)
from fdm.ui.widgets import (
    NoWheelComboBox,
    NoWheelDoubleSpinBox,
    NoWheelSpinBox,
)


_PROFILE_PREVIEW_DEBOUNCE_MS = 150
_PROFILE_PREVIEW_MAX_CROP_PIXELS = 4_000_000
_PROFILE_PREVIEW_MAX_SAMPLES = 2_048
_PROFILE_PREVIEW_MAX_SAMPLE_WORK = 131_072


def _freeze_preview_points(
    points: Iterable[Iterable[object]],
) -> tuple[tuple[float, float], ...]:
    frozen: list[tuple[float, float]] = []
    for raw_point in points:
        coordinates = tuple(raw_point)
        if len(coordinates) != 2:
            raise ValueError("剖面预览坐标必须包含 X、Y")
        x = float(coordinates[0])
        y = float(coordinates[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("剖面预览坐标必须是有限数")
        frozen.append((x, y))
    if frozen and len(frozen) < 2:
        raise ValueError("剖面预览至少需要两个点")
    return tuple(frozen)


@dataclass(frozen=True, slots=True)
class ProfilePreviewContext:
    """Immutable pixels and RAW geometry used only by the parameter preview."""

    plane: RasterPlane
    document_id: str
    calibration: AnalysisCalibrationSnapshot = AnalysisCalibrationSnapshot()
    source_pixel_revision: int = 0
    line_points: tuple[tuple[float, float], ...] = ()
    rectangle_points: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.plane, RasterPlane) or self.plane.is_empty:
            raise ValueError("剖面预览必须包含非空的冻结像素")
        document_id = str(self.document_id).strip()
        if not document_id:
            raise ValueError("剖面预览 document_id 不能为空")
        revision = int(self.source_pixel_revision)
        if revision < 0:
            raise ValueError("剖面预览像素修订号不能为负数")
        if not isinstance(self.calibration, AnalysisCalibrationSnapshot):
            raise TypeError("剖面预览标定快照类型无效")
        object.__setattr__(self, "document_id", document_id)
        object.__setattr__(self, "source_pixel_revision", revision)
        object.__setattr__(
            self,
            "line_points",
            _freeze_preview_points(self.line_points),
        )
        object.__setattr__(
            self,
            "rectangle_points",
            _freeze_preview_points(self.rectangle_points),
        )

    def points_for(
        self,
        aggregation: object,
    ) -> tuple[tuple[float, float], ...]:
        return (
            self.line_points
            if str(aggregation) == "line"
            else self.rectangle_points
        )


class ProfilePreviewCurve(QWidget):
    """Small dependency-free curve view for a transient intensity profile."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._curve: AnalysisCurve | None = None
        self._message = "等待参数…"
        self.setMinimumHeight(140)
        self.setMaximumHeight(210)

    def set_curve(self, curve: AnalysisCurve | None) -> None:
        self._curve = curve
        self._message = ""
        self.update()

    def set_message(self, message: str) -> None:
        self._curve = None
        self._message = str(message)
        self.update()

    def paintEvent(self, event: object) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        palette = self.palette()
        painter.fillRect(self.rect(), palette.color(QPalette.ColorRole.Base))
        plot = QRectF(self.rect()).adjusted(38, 12, -12, -28)
        painter.setPen(QPen(palette.color(QPalette.ColorRole.Mid), 1))
        painter.drawRect(plot)
        curve = self._curve
        if curve is None:
            painter.setPen(palette.color(QPalette.ColorRole.PlaceholderText))
            painter.drawText(
                plot.adjusted(10, 10, -10, -10),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                self._message or "没有可预览的数据。",
            )
            return
        valid = tuple(
            (float(x), float(y))
            for x, y in zip(curve.x, curve.y, strict=True)
            if y is not None and math.isfinite(float(x)) and math.isfinite(float(y))
        )
        if len(valid) < 2:
            painter.setPen(palette.color(QPalette.ColorRole.PlaceholderText))
            painter.drawText(
                plot.adjusted(10, 10, -10, -10),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                "有效采样点不足，无法绘制剖面。",
            )
            return
        minimum_x = min(point[0] for point in valid)
        maximum_x = max(point[0] for point in valid)
        minimum_y = min(point[1] for point in valid)
        maximum_y = max(point[1] for point in valid)
        range_x = maximum_x - minimum_x or 1.0
        range_y = maximum_y - minimum_y or 1.0
        path = QPainterPath()
        started = False
        for raw_x, raw_y in zip(curve.x, curve.y, strict=True):
            if raw_y is None:
                started = False
                continue
            x = float(raw_x)
            y = float(raw_y)
            if not math.isfinite(x) or not math.isfinite(y):
                started = False
                continue
            target_x = plot.left() + ((x - minimum_x) / range_x) * plot.width()
            target_y = plot.bottom() - ((y - minimum_y) / range_y) * plot.height()
            if started:
                path.lineTo(target_x, target_y)
            else:
                path.moveTo(target_x, target_y)
                started = True
        painter.setPen(QPen(QColor("#2A9D8F"), 1.8))
        painter.drawPath(path)
        painter.setPen(palette.color(QPalette.ColorRole.Text))
        painter.drawText(
            QRectF(plot.left(), plot.bottom() + 3, plot.width(), 22),
            Qt.AlignmentFlag.AlignCenter,
            f"距离（{curve.x_unit or 'px'}） / 强度",
        )


def _profile_preview_array(plane: RasterPlane) -> NDArray[np.generic]:
    dtype: np.dtype[object]
    if plane.pixel_type is RasterPixelType.GRAY8:
        dtype = np.dtype(np.uint8)
    elif plane.pixel_type is RasterPixelType.GRAY16:
        dtype = np.dtype("<u2")
    elif plane.pixel_type is RasterPixelType.GRAY32_FLOAT:
        dtype = np.dtype("<f4")
    else:
        dtype = np.dtype(np.uint8)
    array = np.frombuffer(plane.data, dtype=dtype)
    channels = plane.pixel_type.channel_count
    if channels == 1:
        return array.reshape((plane.height, plane.width))
    return array.reshape((plane.height, plane.width, channels))


def _bounded_profile_preview_request(
    request: ImageAnalysisTaskRequest,
) -> ImageAnalysisTaskRequest:
    parameters = request.parameters
    aggregation = str(parameters.get("aggregation", "line"))
    points = _freeze_preview_points(parameters.get("points", ()))
    if len(points) < 2:
        raise ValueError("没有可用于实时预览的线段、折线或矩形 ROI。")
    line_width = float(parameters.get("line_width", 1.0))
    margin = max(2.0, math.ceil(line_width / 2.0) + 2.0)
    if aggregation != "line":
        margin = 1.0
    minimum_x = min(point[0] for point in points)
    maximum_x = max(point[0] for point in points)
    minimum_y = min(point[1] for point in points)
    maximum_y = max(point[1] for point in points)
    left = max(0, int(math.floor(minimum_x - margin)))
    top = max(0, int(math.floor(minimum_y - margin)))
    right = min(request.plane.width, int(math.ceil(maximum_x + margin)) + 1)
    bottom = min(request.plane.height, int(math.ceil(maximum_y + margin)) + 1)
    if right <= left or bottom <= top:
        raise ValueError("当前选择不在冻结的原始像素视场内。")
    crop_pixels = (right - left) * (bottom - top)
    if crop_pixels > _PROFILE_PREVIEW_MAX_CROP_PIXELS:
        raise ValueError(
            "当前选择的实时预览范围超过 400 万像素；"
            "请缩小矩形 ROI，或直接开始正式分析。"
        )
    source = _profile_preview_array(request.plane)
    crop = np.ascontiguousarray(source[top:bottom, left:right])
    crop_plane = RasterPlane(
        width=right - left,
        height=bottom - top,
        pixel_type=request.plane.pixel_type,
        data=crop.tobytes(order="C"),
    )
    shifted_points = tuple((x - left, y - top) for x, y in points)
    requested_spacing = float(parameters.get("sample_spacing", 1.0))
    if aggregation == "line":
        total_length = sum(
            math.hypot(
                shifted_points[index + 1][0] - shifted_points[index][0],
                shifted_points[index + 1][1] - shifted_points[index][1],
            )
            for index in range(len(shifted_points) - 1)
        )
        width_samples = max(1, int(math.ceil(line_width)))
        sample_budget = max(
            16,
            min(
                _PROFILE_PREVIEW_MAX_SAMPLES,
                _PROFILE_PREVIEW_MAX_SAMPLE_WORK // width_samples,
            ),
        )
        preview_spacing = max(
            requested_spacing,
            total_length / max(1, sample_budget - 1),
        )
    else:
        axis_extent = (
            bottom - top
            if aggregation == "rectangle_rows"
            else right - left
        )
        preview_spacing = max(
            requested_spacing,
            axis_extent / max(1, _PROFILE_PREVIEW_MAX_SAMPLES - 1),
        )
    preview_parameters = dict(parameters)
    preview_parameters["points"] = shifted_points
    preview_parameters["sample_spacing"] = preview_spacing
    return ImageAnalysisTaskRequest(
        tool=AnalysisTool.PROFILE,
        request_id=request.request_id,
        generation=request.generation,
        document_id=request.document_id,
        source_pixel_revision=request.source_pixel_revision,
        plane=crop_plane,
        calibration=request.calibration,
        parameters=preview_parameters,
    )


def execute_profile_preview_task(
    request: ImageAnalysisTaskRequest,
    token: CancellationToken,
    phase_callback: AnalysisPhaseCallback,
) -> ImageAnalysisTaskResult:
    """Run a profile preview on a bounded crop, never on the whole source."""

    token.raise_if_cancelled()
    bounded_request = _bounded_profile_preview_request(request)
    token.raise_if_cancelled()
    return execute_analysis_task(bounded_request, token, phase_callback)


class AnalysisParameterKind(StrEnum):
    BOOLEAN = "boolean"
    INTEGER = "integer"
    NUMBER = "number"
    TEXT = "text"
    CHOICE = "choice"
    JSON = "json"


@dataclass(frozen=True, slots=True)
class AnalysisParameterField:
    key: str
    chinese_name: str
    kind: AnalysisParameterKind
    default: object = None
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[tuple[str, object], ...] = ()
    description: str = ""
    nullable: bool = False

    def __post_init__(self) -> None:
        if not str(self.key).strip() or not str(self.chinese_name).strip():
            raise ValueError("参数键和中文名称不能为空")
        object.__setattr__(self, "kind", AnalysisParameterKind(self.kind))
        if self.kind is AnalysisParameterKind.CHOICE and not self.choices:
            raise ValueError(f"选择参数 {self.key} 必须提供选项")

    def normalize(self, value: object) -> object:
        if value is None:
            if self.nullable:
                return None
            value = self.default
        if self.kind is AnalysisParameterKind.BOOLEAN:
            return bool(value)
        if self.kind is AnalysisParameterKind.INTEGER:
            if isinstance(value, bool):
                raise TypeError(f"{self.chinese_name} 必须是整数")
            normalized: object = int(value)
        elif self.kind is AnalysisParameterKind.NUMBER:
            if isinstance(value, bool):
                raise TypeError(f"{self.chinese_name} 必须是数值")
            normalized = float(value)
            if not math.isfinite(normalized):
                raise ValueError(f"{self.chinese_name} 必须是有限数")
        elif self.kind is AnalysisParameterKind.CHOICE:
            allowed = tuple(item for _label, item in self.choices)
            if value not in allowed:
                raise ValueError(f"{self.chinese_name} 的选项无效")
            return value
        elif self.kind is AnalysisParameterKind.JSON:
            if isinstance(value, str):
                try:
                    normalized = json.loads(value)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{self.chinese_name} 不是有效 JSON") from exc
            else:
                normalized = value
            encoded = json.dumps(normalized, allow_nan=False)
            return json.loads(encoded)
        else:
            return str(value)
        numeric = float(normalized)
        if self.minimum is not None and numeric < self.minimum:
            raise ValueError(f"{self.chinese_name} 不能小于 {self.minimum:g}")
        if self.maximum is not None and numeric > self.maximum:
            raise ValueError(f"{self.chinese_name} 不能大于 {self.maximum:g}")
        return normalized


@dataclass(frozen=True, slots=True)
class AnalysisParameterSchema:
    tool: AnalysisTool
    chinese_name: str
    version: str
    fields: tuple[AnalysisParameterField, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool", AnalysisTool(self.tool))
        if len({field.key for field in self.fields}) != len(self.fields):
            raise ValueError("参数 schema 不能包含重复键")

    def defaults(self) -> dict[str, object]:
        return {
            field.key: field.normalize(field.default)
            for field in self.fields
            if field.default is not None or field.nullable
        }

    def validate(
        self,
        values: Mapping[str, object],
        *,
        include_defaults: bool = True,
    ) -> dict[str, object]:
        unknown = set(values) - {field.key for field in self.fields}
        if unknown:
            raise ValueError(f"包含未知参数：{', '.join(sorted(unknown))}")
        normalized = self.defaults() if include_defaults else {}
        for field in self.fields:
            if field.key in values:
                normalized[field.key] = field.normalize(values[field.key])
        return normalized

    def to_json_schema(self) -> dict[str, object]:
        properties: dict[str, object] = {}
        type_names = {
            AnalysisParameterKind.BOOLEAN: "boolean",
            AnalysisParameterKind.INTEGER: "integer",
            AnalysisParameterKind.NUMBER: "number",
            AnalysisParameterKind.TEXT: "string",
            AnalysisParameterKind.CHOICE: None,
            AnalysisParameterKind.JSON: None,
        }
        for field in self.fields:
            property_schema: dict[str, object] = {
                "title": field.chinese_name,
                "description": field.description,
                "default": field.default,
            }
            type_name = type_names[field.kind]
            if type_name is not None:
                property_schema["type"] = (
                    [type_name, "null"] if field.nullable else type_name
                )
            if field.choices:
                property_schema["enum"] = [
                    value for _label, value in field.choices
                ] + ([None] if field.nullable else [])
            if field.minimum is not None:
                property_schema["minimum"] = field.minimum
            if field.maximum is not None:
                property_schema["maximum"] = field.maximum
            properties[field.key] = property_schema
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": f"fdm.{self.tool.value}.parameters.v{self.version}",
            "title": self.chinese_name,
            "type": "object",
            "additionalProperties": False,
            "properties": properties,
        }


CHANNELS = (
    ("加权亮度", "luminance"),
    ("红色", "red"),
    ("绿色", "green"),
    ("蓝色", "blue"),
)
RGB_CHANNELS = CHANNELS + (("RGB 三通道统计", "rgb"),)
FOREGROUND = (("亮前景", "bright"), ("暗前景", "dark"))


def _field(
    key: str,
    name: str,
    kind: AnalysisParameterKind,
    default: object = None,
    **kwargs: object,
) -> AnalysisParameterField:
    return AnalysisParameterField(key, name, kind, default, **kwargs)


def _channel(*, rgb: bool = False) -> AnalysisParameterField:
    return _field(
        "channel",
        "分析通道",
        AnalysisParameterKind.CHOICE,
        "luminance",
        choices=RGB_CHANNELS if rgb else CHANNELS,
    )


def _binary_fields() -> tuple[AnalysisParameterField, ...]:
    return (
        _channel(),
        _field("threshold", "二值阈值", AnalysisParameterKind.NUMBER, 128.0),
        _field(
            "foreground",
            "前景极性",
            AnalysisParameterKind.CHOICE,
            "bright",
            choices=FOREGROUND,
        ),
    )


ANALYSIS_PARAMETER_SCHEMAS: Mapping[
    AnalysisTool,
    AnalysisParameterSchema,
] = MappingProxyType(
    {
        AnalysisTool.SHAPE: AnalysisParameterSchema(
            AnalysisTool.SHAPE, "形状测量参数", "2", ()
        ),
        AnalysisTool.INTENSITY: AnalysisParameterSchema(
            AnalysisTool.INTENSITY,
            "灰度与颜色统计参数",
            "2",
            (
                _channel(rgb=True),
                _field(
                    "percentile_levels",
                    "分位数列表",
                    AnalysisParameterKind.JSON,
                    [10, 25, 50, 75, 90],
                ),
                _field(
                    "threshold_low",
                    "阈值下限",
                    AnalysisParameterKind.NUMBER,
                    None,
                    nullable=True,
                ),
                _field(
                    "threshold_high",
                    "阈值上限",
                    AnalysisParameterKind.NUMBER,
                    None,
                    nullable=True,
                ),
            ),
        ),
        AnalysisTool.HISTOGRAM: AnalysisParameterSchema(
            AnalysisTool.HISTOGRAM,
            "直方图参数",
            "2",
            (
                _channel(),
                _field(
                    "bins",
                    "分箱数量",
                    AnalysisParameterKind.INTEGER,
                    256,
                    minimum=1,
                    maximum=65536,
                ),
                _field(
                    "value_range",
                    "固定范围 [下限, 上限]",
                    AnalysisParameterKind.JSON,
                    None,
                    nullable=True,
                ),
                _field(
                    "log_counts",
                    "对数频数显示",
                    AnalysisParameterKind.BOOLEAN,
                    False,
                ),
            ),
        ),
        AnalysisTool.FFT_POWER_SPECTRUM: AnalysisParameterSchema(
            AnalysisTool.FFT_POWER_SPECTRUM,
            "FFT 功率谱参数",
            "1",
            (
                _channel(),
                _field(
                    "logarithmic",
                    "对数功率",
                    AnalysisParameterKind.BOOLEAN,
                    True,
                ),
                _field(
                    "centered",
                    "零频居中",
                    AnalysisParameterKind.BOOLEAN,
                    True,
                ),
                _field(
                    "window",
                    "窗函数",
                    AnalysisParameterKind.CHOICE,
                    "none",
                    choices=(("无", "none"), ("Tukey", "tukey")),
                ),
                _field(
                    "tukey_alpha",
                    "Tukey alpha",
                    AnalysisParameterKind.NUMBER,
                    0.25,
                    minimum=0.0,
                    maximum=1.0,
                ),
            ),
        ),
        AnalysisTool.PROFILE: AnalysisParameterSchema(
            AnalysisTool.PROFILE,
            "强度剖面参数",
            "2",
            (
                _field("points", "采样点 / 矩形角点", AnalysisParameterKind.JSON, []),
                _field(
                    "aggregation",
                    "采样方式",
                    AnalysisParameterKind.CHOICE,
                    "line",
                    choices=(
                        ("沿线宽度平均", "line"),
                        ("矩形逐行平均", "rectangle_rows"),
                        ("矩形逐列平均", "rectangle_columns"),
                    ),
                ),
                _field(
                    "line_width",
                    "线宽（像素）",
                    AnalysisParameterKind.NUMBER,
                    1.0,
                    minimum=0.01,
                ),
                _field(
                    "sample_spacing",
                    "采样间距（像素）",
                    AnalysisParameterKind.NUMBER,
                    1.0,
                    minimum=0.01,
                ),
                _channel(),
            ),
        ),
        AnalysisTool.PARTICLES: AnalysisParameterSchema(
            AnalysisTool.PARTICLES,
            "粒子分析参数",
            "2",
            _binary_fields()
            + (
                _field(
                    "connectivity",
                    "连通性",
                    AnalysisParameterKind.CHOICE,
                    8,
                    choices=(("4 邻域", 4), ("8 邻域", 8)),
                ),
                _field(
                    "min_area_px",
                    "最小面积（px²）",
                    AnalysisParameterKind.INTEGER,
                    1,
                    minimum=1,
                ),
                _field(
                    "max_area_px",
                    "最大面积（px²）",
                    AnalysisParameterKind.INTEGER,
                    None,
                    minimum=1,
                    nullable=True,
                ),
                _field(
                    "min_circularity",
                    "最小圆度",
                    AnalysisParameterKind.NUMBER,
                    0.0,
                    minimum=0,
                    maximum=1,
                ),
                _field(
                    "max_circularity",
                    "最大圆度",
                    AnalysisParameterKind.NUMBER,
                    1.0,
                    minimum=0,
                    maximum=1,
                ),
                _field("include_holes", "填充孔洞", AnalysisParameterKind.BOOLEAN, False),
                _field("exclude_edge", "排除触边粒子", AnalysisParameterKind.BOOLEAN, False),
                _field("watershed", "启用 Watershed 分离", AnalysisParameterKind.BOOLEAN, False),
                _field(
                    "watershed_min_distance",
                    "Watershed 峰最小距离",
                    AnalysisParameterKind.INTEGER,
                    3,
                    minimum=1,
                ),
            ),
        ),
        AnalysisTool.MAXIMA: AnalysisParameterSchema(
            AnalysisTool.MAXIMA,
            "极值检测参数",
            "2",
            (
                _channel(),
                _field(
                    "algorithm_version",
                    "Prominence 定义",
                    AnalysisParameterKind.CHOICE,
                    "1",
                    choices=(
                        ("v1 局部邻域 prominence", "1"),
                        ("v2 地形 prominence", "2"),
                    ),
                ),
                _field(
                    "minimum_value",
                    "最小峰值",
                    AnalysisParameterKind.NUMBER,
                    None,
                    nullable=True,
                ),
                _field(
                    "prominence",
                    "最小 prominence",
                    AnalysisParameterKind.NUMBER,
                    0.0,
                    minimum=0,
                ),
                _field(
                    "neighborhood_radius",
                    "局部邻域半径",
                    AnalysisParameterKind.INTEGER,
                    1,
                    minimum=1,
                ),
                _field(
                    "min_distance",
                    "极值最小间距",
                    AnalysisParameterKind.NUMBER,
                    1.0,
                    minimum=0.01,
                ),
                _field("exclude_edge", "排除边缘极值", AnalysisParameterKind.BOOLEAN, False),
                _field(
                    "max_points",
                    "最大点数",
                    AnalysisParameterKind.INTEGER,
                    None,
                    minimum=1,
                    nullable=True,
                ),
            ),
        ),
        AnalysisTool.DIRECTIONALITY: AnalysisParameterSchema(
            AnalysisTool.DIRECTIONALITY,
            "方向性参数",
            "2",
            (
                _channel(),
                _field(
                    "algorithm_version",
                    "算法版本",
                    AnalysisParameterKind.CHOICE,
                    2,
                    choices=(
                        ("v2：5×5 梯度与 Fourier 融合", 2),
                        ("v1：历史结果", 1),
                    ),
                ),
                _field("bins", "方向分箱数", AnalysisParameterKind.INTEGER, 180, minimum=4),
                _field("gradient_sigma", "梯度平滑 Sigma", AnalysisParameterKind.NUMBER, 1.0, minimum=0),
                _field("minimum_gradient", "最小梯度", AnalysisParameterKind.NUMBER, 0.0, minimum=0),
                _field(
                    "histogram_smoothing_bins",
                    "直方图平滑宽度",
                    AnalysisParameterKind.NUMBER,
                    1.0,
                    minimum=0,
                ),
                _field("peak_min_fraction", "峰相对阈值", AnalysisParameterKind.NUMBER, 0.1, minimum=0, maximum=1),
                _field("max_peaks", "最大方向峰数", AnalysisParameterKind.INTEGER, 8, minimum=1),
            ),
        ),
        AnalysisTool.SKELETON: AnalysisParameterSchema(
            AnalysisTool.SKELETON,
            "骨架网络参数",
            "2",
            _binary_fields()
            + (
                _field(
                    "algorithm_version",
                    "算法版本",
                    AnalysisParameterKind.CHOICE,
                    2,
                    choices=(
                        ("v2：节点分类与可审计剪枝", 2),
                        ("v1：历史结果", 1),
                    ),
                ),
                _field(
                    "already_skeletonized",
                    "输入已骨架化",
                    AnalysisParameterKind.BOOLEAN,
                    False,
                ),
                _field(
                    "prune_terminal_branches_below",
                    "末端分支剪枝阈值",
                    AnalysisParameterKind.NUMBER,
                    0.0,
                    minimum=0,
                ),
            ),
        ),
        AnalysisTool.LOCAL_THICKNESS: AnalysisParameterSchema(
            AnalysisTool.LOCAL_THICKNESS,
            "局部厚度参数",
            "2",
            _binary_fields(),
        ),
        AnalysisTool.TUBENESS: AnalysisParameterSchema(
            AnalysisTool.TUBENESS,
            "Tubeness 参数",
            "1",
            (
                _channel(),
                _field("scales", "尺度列表", AnalysisParameterKind.JSON, [1.0, 2.0, 4.0]),
                _field("beta", "Beta", AnalysisParameterKind.NUMBER, 0.5, minimum=0.000001),
                _field(
                    "structure_scale",
                    "结构尺度",
                    AnalysisParameterKind.NUMBER,
                    None,
                    minimum=0.000001,
                    nullable=True,
                ),
                _field("bright_ridges", "检测亮脊线", AnalysisParameterKind.BOOLEAN, True),
            ),
        ),
        AnalysisTool.GLCM: AnalysisParameterSchema(
            AnalysisTool.GLCM,
            "GLCM 参数",
            "2",
            (
                _channel(),
                _field("levels", "量化级数", AnalysisParameterKind.INTEGER, 32, minimum=2, maximum=256),
                _field("distances", "距离列表", AnalysisParameterKind.JSON, [1]),
                _field("directions_degrees", "方向列表（°）", AnalysisParameterKind.JSON, [0, 45, 90, 135]),
                _field("value_range", "量化范围", AnalysisParameterKind.JSON, None, nullable=True),
                _field("symmetric", "对称 GLCM", AnalysisParameterKind.BOOLEAN, True),
            ),
        ),
        AnalysisTool.SPATIAL_DISTRIBUTION: AnalysisParameterSchema(
            AnalysisTool.SPATIAL_DISTRIBUTION,
            "空间分布参数",
            "2",
            (
                _field("points", "点集", AnalysisParameterKind.JSON, []),
                _field(
                    "algorithm_version",
                    "算法版本",
                    AnalysisParameterKind.CHOICE,
                    2,
                    choices=(
                        ("v2：Ripley K/L 平移边界校正", 2),
                        ("v1：历史最近邻与空间密度", 1),
                    ),
                ),
                _field(
                    "study_area",
                    "研究区域面积",
                    AnalysisParameterKind.NUMBER,
                    None,
                    minimum=0.000001,
                    nullable=True,
                ),
                _field(
                    "study_bounds",
                    "矩形研究区域边界",
                    AnalysisParameterKind.JSON,
                    None,
                    nullable=True,
                ),
                _field(
                    "ripley_radii",
                    "Ripley 半径列表",
                    AnalysisParameterKind.JSON,
                    [],
                ),
                _field("point_scope", "点集范围", AnalysisParameterKind.TEXT, "all"),
                _field("point_group_id", "点类别 ID", AnalysisParameterKind.TEXT, ""),
                _field("point_group_label", "点类别名称", AnalysisParameterKind.TEXT, ""),
                _field("study_area_mode", "研究区域来源", AnalysisParameterKind.TEXT, "scope"),
            ),
        ),
        AnalysisTool.SURFACE: AnalysisParameterSchema(
            AnalysisTool.SURFACE,
            "二维强度表面参数",
            "1",
            (
                _channel(),
                _field("sample_step_x", "X 采样步长", AnalysisParameterKind.INTEGER, 1, minimum=1),
                _field("sample_step_y", "Y 采样步长", AnalysisParameterKind.INTEGER, 1, minimum=1),
            ),
        ),
    }
)


def analysis_parameter_schema(
    tool: AnalysisTool | str,
) -> AnalysisParameterSchema:
    return ANALYSIS_PARAMETER_SCHEMAS[AnalysisTool(tool)]


class AnalysisParametersDialog(QDialog):
    """A compact editor backed by the same immutable parameter schema."""

    def __init__(
        self,
        tool: AnalysisTool | str,
        parameters: Mapping[str, object] | None = None,
        *,
        profile_store: AnalysisMeasurementProfileStore | None = None,
        profile_preview_context: ProfilePreviewContext | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.schema = analysis_parameter_schema(tool)
        self.setWindowTitle(self.schema.chinese_name)
        self.setModal(True)
        self._editors: dict[str, QWidget] = {}
        self._profile_preview_context = (
            profile_preview_context
            if self.schema.tool is AnalysisTool.PROFILE
            else None
        )
        self._profile_preview_controller: ImageAnalysisTaskController | None = None
        self._profile_preview_timer: QTimer | None = None
        self._profile_preview_curve: ProfilePreviewCurve | None = None
        self._profile_preview_status: QLabel | None = None
        self._profile_preview_request_id: str | None = None
        self._profile_preview_generation = 0
        self._profile_preview_closed = False
        normalized = self.schema.validate(parameters or {})

        root = QVBoxLayout(self)
        explanation = QLabel(
            f"{self.schema.tool.chinese_name} · 参数 schema v{self.schema.version}",
            self,
        )
        root.addWidget(explanation)
        self.output_field_selector = AnalysisOutputFieldSelector(
            f"fdm.{self.schema.tool.value}",
            parent=self,
        )
        self.profile_controls = AnalysisProfileControls(
            tool_id=f"fdm.{self.schema.tool.value}",
            tool_version=self.schema.version,
            read_parameters=self.parameters,
            apply_parameters=self.set_parameters,
            read_output_fields=self.output_fields,
            apply_output_fields=self.set_output_fields,
            store=profile_store,
            parent=self,
        )
        root.addWidget(self.profile_controls)
        form = QFormLayout()
        for field in self.schema.fields:
            editor = self._create_editor(field, normalized.get(field.key))
            if field.description:
                editor.setToolTip(field.description)
            self._editors[field.key] = editor
            form.addRow(field.chinese_name, editor)
        root.addLayout(form)
        root.addWidget(self.output_field_selector)
        if self.schema.tool is AnalysisTool.PROFILE:
            self._create_profile_preview(root)
        self._error_label = QLabel("", self)
        self._error_label.setWordWrap(True)
        self._error_label.setStyleSheet("color: #b3261e;")
        root.addWidget(self._error_label)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self.finished.connect(self._shutdown_profile_preview)

    def accept(self) -> None:
        try:
            self.parameters()
        except (TypeError, ValueError) as exc:
            self._error_label.setText(str(exc))
            return
        self._error_label.clear()
        super().accept()

    def parameters(self) -> dict[str, object]:
        raw = {
            field.key: self._editor_value(field, self._editors[field.key])
            for field in self.schema.fields
        }
        return self.schema.validate(raw)

    def set_parameters(self, values: Mapping[str, object]) -> None:
        normalized = self.schema.validate(values)
        for field in self.schema.fields:
            self._set_editor_value(
                field,
                self._editors[field.key],
                normalized.get(field.key),
            )

    def _create_profile_preview(self, root: QVBoxLayout) -> None:
        self._profile_preview_status = QLabel("实时剖面预览：等待参数…", self)
        self._profile_preview_status.setWordWrap(True)
        root.addWidget(self._profile_preview_status)
        self._profile_preview_curve = ProfilePreviewCurve(self)
        root.addWidget(self._profile_preview_curve)
        self._profile_preview_controller = ImageAnalysisTaskController(
            executor=execute_profile_preview_task,
            parent=self,
        )
        self._profile_preview_controller.analysisReady.connect(
            self._on_profile_preview_ready
        )
        self._profile_preview_controller.taskFailed.connect(
            self._on_profile_preview_failed
        )
        self._profile_preview_timer = QTimer(self)
        self._profile_preview_timer.setSingleShot(True)
        self._profile_preview_timer.setInterval(_PROFILE_PREVIEW_DEBOUNCE_MS)
        self._profile_preview_timer.timeout.connect(self._start_profile_preview)
        for editor in self._editors.values():
            self._connect_preview_editor(editor)
        self._profile_preview_timer.start()

    def _connect_preview_editor(self, editor: QWidget) -> None:
        if isinstance(editor, QCheckBox):
            editor.toggled.connect(self._schedule_profile_preview)
        elif isinstance(editor, NoWheelSpinBox):
            editor.valueChanged.connect(self._schedule_profile_preview)
        elif isinstance(editor, NoWheelDoubleSpinBox):
            editor.valueChanged.connect(self._schedule_profile_preview)
        elif isinstance(editor, NoWheelComboBox):
            editor.currentIndexChanged.connect(self._schedule_profile_preview)
        elif isinstance(editor, QLineEdit):
            editor.textChanged.connect(self._schedule_profile_preview)

    def _schedule_profile_preview(self, *args: object) -> None:
        del args
        if (
            self._profile_preview_closed
            or self._profile_preview_timer is None
        ):
            return
        self._profile_preview_timer.start(_PROFILE_PREVIEW_DEBOUNCE_MS)

    def _set_profile_preview_message(self, message: str) -> None:
        if self._profile_preview_status is not None:
            self._profile_preview_status.setText(str(message))
        if self._profile_preview_curve is not None:
            self._profile_preview_curve.set_message(str(message))

    def _start_profile_preview(self) -> None:
        if self._profile_preview_closed:
            return
        controller = self._profile_preview_controller
        context = self._profile_preview_context
        self._profile_preview_generation += 1
        self._profile_preview_request_id = None
        if controller is None or context is None:
            self._set_profile_preview_message(
                "实时剖面预览不可用：没有冻结的当前图片与选择。"
            )
            return
        try:
            parameters = self.parameters()
        except (TypeError, ValueError) as exc:
            controller.cancel()
            self._set_profile_preview_message(f"参数尚未完成：{exc}")
            return
        aggregation = str(parameters.get("aggregation", "line"))
        points = context.points_for(aggregation)
        if len(points) < 2:
            controller.cancel()
            selection_name = (
                "线段或折线"
                if aggregation == "line"
                else "矩形 ROI"
            )
            self._set_profile_preview_message(
                f"没有可预览的{selection_name}；请先在画布中选择。"
            )
            return
        parameters["points"] = points
        self._set_profile_preview_message("正在计算实时剖面预览…")
        try:
            request = controller.start(
                tool=AnalysisTool.PROFILE,
                document_id=context.document_id,
                source_pixel_revision=context.source_pixel_revision,
                plane=context.plane,
                calibration=context.calibration,
                parameters=parameters,
            )
        except (MemoryError, TypeError, ValueError) as exc:
            self._set_profile_preview_message(f"无法开始实时预览：{exc}")
            return
        self._profile_preview_request_id = request.request_id
        self._profile_preview_generation = request.generation

    def _on_profile_preview_ready(self, result: object) -> None:
        if self._profile_preview_closed or not isinstance(
            result,
            ImageAnalysisTaskResult,
        ):
            return
        if (
            result.tool is not AnalysisTool.PROFILE
            or result.request_id != self._profile_preview_request_id
            or result.generation != self._profile_preview_generation
        ):
            return
        curve = result.curves[0] if result.curves else None
        if curve is None:
            self._set_profile_preview_message(
                "实时预览没有足够的有效采样数据。"
            )
            return
        if self._profile_preview_curve is not None:
            self._profile_preview_curve.set_curve(curve)
        valid_count = int(result.scalars.get("valid_sample_count", 0) or 0)
        sample_count = int(result.scalars.get("sample_count", 0) or 0)
        requested_spacing = float(self.parameters().get("sample_spacing", 1.0))
        effective_spacing = float(
            result.parameters.get("sample_spacing", requested_spacing)
        )
        note = (
            f"；为保持流畅，预览间距临时调整为 {effective_spacing:g}px，"
            "正式分析仍使用设置值"
            if effective_spacing > requested_spacing + 1e-9
            else ""
        )
        if self._profile_preview_status is not None:
            self._profile_preview_status.setText(
                f"实时剖面预览：有效 {valid_count}/{sample_count} 点{note}"
            )

    def _on_profile_preview_failed(
        self,
        request_id: str,
        message: str,
    ) -> None:
        if (
            self._profile_preview_closed
            or str(request_id) != self._profile_preview_request_id
        ):
            return
        self._set_profile_preview_message(f"实时预览失败：{message}")

    def _shutdown_profile_preview(self, *args: object) -> None:
        del args
        if self._profile_preview_closed:
            return
        self._profile_preview_closed = True
        self._profile_preview_generation += 1
        self._profile_preview_request_id = None
        if self._profile_preview_timer is not None:
            self._profile_preview_timer.stop()
        if self._profile_preview_controller is not None:
            self._profile_preview_controller.close()

    def output_fields(self) -> tuple[str, ...] | None:
        return self.output_field_selector.output_fields()

    def set_output_fields(self, fields: Iterable[str] | None) -> None:
        self.output_field_selector.set_output_fields(fields)

    def _create_editor(
        self,
        field: AnalysisParameterField,
        value: object,
    ) -> QWidget:
        if field.kind is AnalysisParameterKind.BOOLEAN:
            editor = QCheckBox(self)
            editor.setChecked(bool(value))
            return editor
        if field.kind is AnalysisParameterKind.INTEGER and not field.nullable:
            editor = NoWheelSpinBox(self)
            editor.setRange(
                int(field.minimum if field.minimum is not None else -2_147_483_648),
                int(field.maximum if field.maximum is not None else 2_147_483_647),
            )
            editor.setValue(int(value))
            return editor
        if field.kind is AnalysisParameterKind.NUMBER and not field.nullable:
            editor = NoWheelDoubleSpinBox(self)
            editor.setDecimals(6)
            editor.setRange(
                field.minimum if field.minimum is not None else -1e15,
                field.maximum if field.maximum is not None else 1e15,
            )
            editor.setValue(float(value))
            return editor
        if field.kind is AnalysisParameterKind.CHOICE:
            editor = NoWheelComboBox(self)
            for label, option in field.choices:
                editor.addItem(label, option)
            index = editor.findData(value)
            editor.setCurrentIndex(max(0, index))
            return editor
        editor = QLineEdit(self)
        if (
            self.schema.tool is AnalysisTool.PROFILE
            and field.key == "points"
        ):
            editor.setReadOnly(True)
            editor.setPlaceholderText("由当前选中的 RAW 线段、折线或矩形 ROI 冻结")
            editor.setToolTip(
                "采样几何来自打开参数窗口时的当前选择，不能在此手工修改。"
            )
        if value is None:
            editor.setText("")
        elif field.kind is AnalysisParameterKind.JSON:
            editor.setText(
                json.dumps(value, ensure_ascii=False, allow_nan=False)
            )
        else:
            editor.setText(str(value))
        return editor

    @staticmethod
    def _editor_value(
        field: AnalysisParameterField,
        editor: QWidget,
    ) -> object:
        if isinstance(editor, QCheckBox):
            return editor.isChecked()
        if isinstance(editor, NoWheelSpinBox):
            return editor.value()
        if isinstance(editor, NoWheelDoubleSpinBox):
            return editor.value()
        if isinstance(editor, NoWheelComboBox):
            return editor.currentData()
        if isinstance(editor, QLineEdit):
            text = editor.text().strip()
            if not text and field.nullable:
                return None
            return text
        raise TypeError("不支持的参数控件")

    @staticmethod
    def _set_editor_value(
        field: AnalysisParameterField,
        editor: QWidget,
        value: object,
    ) -> None:
        if isinstance(editor, QCheckBox):
            editor.setChecked(bool(value))
            return
        if isinstance(editor, NoWheelSpinBox):
            editor.setValue(int(value))
            return
        if isinstance(editor, NoWheelDoubleSpinBox):
            editor.setValue(float(value))
            return
        if isinstance(editor, NoWheelComboBox):
            index = editor.findData(value)
            if index < 0:
                raise ValueError(f"{field.chinese_name} 的预设值不受支持")
            editor.setCurrentIndex(index)
            return
        if isinstance(editor, QLineEdit):
            if value is None:
                editor.clear()
            elif field.kind is AnalysisParameterKind.JSON:
                editor.setText(
                    json.dumps(value, ensure_ascii=False, allow_nan=False)
                )
            else:
                editor.setText(str(value))
            return
        raise TypeError("不支持的参数控件")


__all__ = [
    "ANALYSIS_PARAMETER_SCHEMAS",
    "AnalysisParameterField",
    "AnalysisParameterKind",
    "AnalysisParameterSchema",
    "AnalysisParametersDialog",
    "ProfilePreviewContext",
    "ProfilePreviewCurve",
    "analysis_parameter_schema",
    "execute_profile_preview_task",
]
