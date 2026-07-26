"""Non-modal result browser for independent image-analysis artifacts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math
from pathlib import Path
from threading import Event
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QObject, QRectF, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtGui import (
    QColor,
    QImage,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fdm.analysis_artifacts import (
    AnalysisArtifact,
    AnalysisArtifactStatus,
    AnalysisAssetReference,
    AnalysisAssetKind,
    AnalysisCurve,
    AnalysisObjectKind,
)
from fdm.services.analysis_asset_io import validate_analysis_asset_reference
from fdm.ui.widgets import NoWheelComboBox


NameMap: TypeAlias = Mapping[str, str] | None
_PREVIEW_MAX_ARCHIVE_BYTES = 64 << 20
_PREVIEW_MAX_UNCOMPRESSED_BYTES = 128 << 20
_PREVIEW_MAX_ELEMENTS = 16_000_000
_PREVIEW_MAX_SIDE = 640
_PREVIEW_SCHEMA_MEMBERS = {
    "fdm.skeleton-network.v1": ("skeleton", "skeleton"),
    "fdm.local-thickness.v1": ("thickness_px", "heatmap"),
    "fdm.tubeness.v1": ("response", "heatmap"),
    "fdm.glcm-matrices.v1": ("matrices", "heatmap"),
    "fdm.intensity-surface.v1": ("z", "heatmap"),
}


@dataclass(frozen=True, slots=True)
class AnalysisLocateRequest:
    artifact_id: str
    document_id: str
    object_kind: str | None
    object_id: str | None


@dataclass(frozen=True, slots=True)
class AnalysisActionRequest:
    artifact_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AnalysisExportRequest:
    artifact_ids: tuple[str, ...]
    selected_table_name: str | None


@dataclass(frozen=True, slots=True)
class _AssetPreviewResult:
    generation: int
    artifact_id: str
    asset_path: str
    rgb: NDArray[np.uint8]
    description: str


@dataclass(frozen=True, slots=True)
class _AssetPreviewFailure:
    generation: int
    artifact_id: str
    asset_path: str
    message: str


class _AssetPreviewSignals(QObject):
    ready = Signal(object)
    failed = Signal(object)
    finished = Signal(int)


class _AssetPreviewTask(QRunnable):
    def __init__(
        self,
        *,
        generation: int,
        artifact_id: str,
        candidate: Path,
        reference: AnalysisAssetReference,
    ) -> None:
        super().__init__()
        self.generation = generation
        self.artifact_id = artifact_id
        self.candidate = candidate
        self.reference = reference
        self.signals = _AssetPreviewSignals()
        self._cancelled = Event()

    def cancel(self) -> None:
        self._cancelled.set()

    def run(self) -> None:
        try:
            if self._cancelled.is_set():
                return
            rgb, description = _load_bounded_asset_preview(
                self.candidate,
                self.reference,
            )
            if self._cancelled.is_set():
                return
        except Exception as exc:  # noqa: BLE001 - boundary reports safe text
            if not self._cancelled.is_set():
                self.signals.failed.emit(
                    _AssetPreviewFailure(
                        generation=self.generation,
                        artifact_id=self.artifact_id,
                        asset_path=self.reference.path,
                        message=str(exc),
                    )
                )
            return
        else:
            if not self._cancelled.is_set():
                self.signals.ready.emit(
                    _AssetPreviewResult(
                        generation=self.generation,
                        artifact_id=self.artifact_id,
                        asset_path=self.reference.path,
                        rgb=rgb,
                        description=description,
                    )
                )
        finally:
            self.signals.finished.emit(self.generation)


def _load_bounded_asset_preview(
    candidate: Path,
    reference: AnalysisAssetReference,
) -> tuple[NDArray[np.uint8], str]:
    """Load one known NPZ preview after enforcing a small UI-only budget."""

    schema = str(reference.metadata.get("schema", ""))
    member_spec = _PREVIEW_SCHEMA_MEMBERS.get(schema)
    if member_spec is None:
        raise ValueError("该分析资产暂无内置可视化预览。")
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError("分析资产尚未保存或文件已经缺失。")
    if candidate.stat().st_size > _PREVIEW_MAX_ARCHIVE_BYTES:
        raise ValueError("分析资产超过 64 MiB 预览上限；仍可正常导出原始结果。")
    declared_bytes = _declared_asset_bytes(reference)
    if declared_bytes > _PREVIEW_MAX_UNCOMPRESSED_BYTES:
        raise ValueError("分析资产解压后超过 128 MiB 预览上限。")
    validate_analysis_asset_reference(candidate, reference)
    member_name, render_mode = member_spec
    with np.load(candidate, allow_pickle=False) as archive:
        if member_name not in archive.files:
            raise ValueError(f"分析资产缺少预览成员：{member_name}")
        source = np.asarray(archive[member_name])
    if source.dtype.hasobject or source.dtype.kind not in "biufc":
        raise ValueError("分析资产预览成员使用了不安全的数据类型。")
    if source.size > _PREVIEW_MAX_ELEMENTS:
        raise ValueError("分析资产预览成员超过 1600 万个元素的安全上限。")
    if source.ndim == 3:
        if source.shape[0] < 1:
            raise ValueError("分析资产预览数组为空。")
        source = source[0]
    if source.ndim != 2 or min(source.shape) < 1:
        raise ValueError("分析资产预览成员必须是非空二维数组。")
    sampled, sample_note = _downsample_preview_array(source)
    if render_mode == "skeleton":
        rgb = _render_skeleton_preview(sampled)
        label = "骨架网络"
    else:
        rgb = _render_heatmap_preview(sampled)
        label = {
            "fdm.local-thickness.v1": "局部厚度热力图",
            "fdm.tubeness.v1": "Tubeness 响应热力图",
            "fdm.glcm-matrices.v1": "第一组 GLCM 热力图",
            "fdm.intensity-surface.v1": "二维强度表面热力图",
        }[schema]
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    rgb.setflags(write=False)
    return rgb, f"{label} · {source.shape[1]}×{source.shape[0]}{sample_note}"


def _declared_asset_bytes(reference: AnalysisAssetReference) -> int:
    members = reference.metadata.get("members")
    if not isinstance(members, Mapping):
        raise ValueError("分析资产缺少成员清单，无法安全预览。")
    total = 0
    for raw_descriptor in members.values():
        if not isinstance(raw_descriptor, Mapping):
            raise ValueError("分析资产成员描述不合法。")
        dtype = np.dtype(str(raw_descriptor.get("dtype", "")))
        if dtype.hasobject or dtype.kind not in "biufc":
            raise ValueError("分析资产成员清单包含不安全 dtype。")
        shape = raw_descriptor.get("shape")
        if not isinstance(shape, list) or any(
            isinstance(value, bool) or int(value) < 0 for value in shape
        ):
            raise ValueError("分析资产成员清单包含不合法 shape。")
        elements = math.prod(int(value) for value in shape)
        total += elements * dtype.itemsize
        if total > _PREVIEW_MAX_UNCOMPRESSED_BYTES:
            return total
    return total


def _downsample_preview_array(
    source: NDArray[np.generic],
) -> tuple[NDArray[np.generic], str]:
    height, width = source.shape
    stride = max(1, math.ceil(max(height, width) / _PREVIEW_MAX_SIDE))
    sampled = source[::stride, ::stride]
    return sampled, "" if stride == 1 else f" · 预览按 {stride}× 抽样"


def _render_skeleton_preview(
    source: NDArray[np.generic],
) -> NDArray[np.uint8]:
    mask = np.asarray(source, dtype=bool)
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[..., :] = (17, 24, 39)
    rgb[mask] = (45, 212, 191)
    return rgb


def _render_heatmap_preview(
    source: NDArray[np.generic],
) -> NDArray[np.uint8]:
    values = np.asarray(source, dtype=np.float64)
    finite = np.isfinite(values)
    normalized = np.zeros(values.shape, dtype=np.float64)
    if np.any(finite):
        selected = values[finite]
        low, high = np.percentile(selected, (2.0, 98.0))
        if not math.isfinite(low) or not math.isfinite(high):
            low, high = float(np.min(selected)), float(np.max(selected))
        span = high - low
        if span <= 0.0:
            normalized[finite] = 0.5
        else:
            normalized[finite] = np.clip(
                (values[finite] - low) / span,
                0.0,
                1.0,
            )
    red = np.clip(1.8 * normalized - 0.35, 0.0, 1.0)
    green = np.clip(1.8 - np.abs(normalized - 0.58) * 3.0, 0.0, 1.0)
    blue = np.clip(1.25 - 1.75 * normalized, 0.0, 1.0)
    rgb = np.stack((red, green, blue), axis=2)
    rgb[~finite] = (0.18, 0.18, 0.18)
    return np.rint(rgb * 255.0).astype(np.uint8)


_TOOL_NAMES = {
    "fdm.shape": "形状测量",
    "fdm.shape_analysis": "形状测量",
    "fdm.intensity": "灰度与颜色统计",
    "fdm.intensity_statistics": "灰度与颜色统计",
    "fdm.histogram": "直方图",
    "fdm.profile": "强度剖面",
    "fdm.intensity_profile": "强度剖面",
    "fdm.particles": "粒子分析",
    "fdm.particle_analysis": "粒子分析",
    "fdm.maxima": "极值检测",
    "fdm.maxima_detection": "极值检测",
    "fdm.directionality": "纤维方向性",
    "fdm.skeleton": "骨架网络",
    "fdm.local_thickness": "局部厚度",
    "fdm.tubeness": "Tubeness",
    "fdm.glcm": "Haralick GLCM 纹理",
    "fdm.spatial_distribution": "最近邻与空间密度",
    "fdm.surface": "二维强度表面",
}

_FIELD_NAMES = {
    "accepted_count": "接受数量",
    "area": "净面积",
    "area_from_exact_mask": "使用精确掩膜面积",
    "area_px": "净面积（px²）",
    "area_source": "研究区域来源",
    "aspect_ratio": "长宽比",
    "bins": "分箱数量",
    "branchpoint_count": "分支点数量",
    "candidate_plateau_count": "候选平台数量",
    "channel": "通道",
    "circularity": "圆度",
    "connected_component_count": "连通分量数量",
    "connectivity": "连通性",
    "convention": "角度约定",
    "coordinate_unit": "坐标单位",
    "definition": "计算定义",
    "ellipse_angle_degrees": "拟合椭圆方向（°）",
    "ellipse_major": "拟合椭圆长轴",
    "ellipse_minor": "拟合椭圆短轴",
    "endpoint_count": "端点数量",
    "equivalent_circle_diameter": "等效圆直径",
    "feret_angle_degrees": "最大 Feret 方向（°）",
    "feret_max": "最大 Feret 直径",
    "feret_min": "最小 Feret 直径",
    "finite_sample_count": "有限样本数",
    "foreground_pixel_count": "前景像素数",
    "hole_area_px": "孔洞面积（px²）",
    "hole_count": "孔洞数量",
    "hole_perimeter": "孔洞周长",
    "hole_perimeter_px": "孔洞周长（px）",
    "include_holes": "包含孔洞",
    "included_pixel_count": "纳入像素数",
    "integrated_density": "积分密度",
    "intensity_centroid_x_px": "强度重心 X（px）",
    "intensity_centroid_y_px": "强度重心 Y（px）",
    "intensity_unit": "强度单位",
    "isolated_point_count": "孤立点数量",
    "levels": "量化级数",
    "loop_count": "环路数量",
    "masked_sample_count": "掩膜内样本数",
    "maximum": "最大值",
    "maximum_geodesic_distance": "最大测地距离",
    "maximum_nearest_neighbor_distance": "最大最近邻距离",
    "maximum_response": "最大响应",
    "maximum_thickness_px": "最大局部厚度（px）",
    "mean": "均值",
    "mean_nearest_neighbor_distance": "平均最近邻距离",
    "mean_thickness_px": "平均局部厚度（px）",
    "median": "中位数",
    "median_nearest_neighbor_distance": "最近邻距离中位数",
    "minimum": "最小值",
    "minimum_nearest_neighbor_distance": "最小最近邻距离",
    "non_finite_count": "非有限像素数",
    "non_finite_pixel_count": "非有限像素数",
    "non_finite_sample_count": "非有限样本数",
    "net_area": "净面积",
    "outer_perimeter": "外轮廓周长",
    "outer_perimeter_px": "外轮廓周长（px）",
    "peak_count": "方向峰数量",
    "point_group_id": "计数点类别 ID",
    "point_group_label": "计数点类别",
    "point_scope": "计数点范围",
    "point_count": "点数量",
    "quantization_maximum": "量化上限",
    "quantization_minimum": "量化下限",
    "rejected_by_area_count": "因面积剔除数量",
    "rejected_by_circularity_count": "因圆度剔除数量",
    "rejected_edge_count": "边缘对象剔除数量",
    "roundness": "Roundness",
    "sample_count": "样本总数",
    "scale_count": "尺度数量",
    "solidity": "Solidity",
    "spatial_density": "空间密度",
    "stddev": "总体标准差",
    "study_area": "研究区域面积",
    "study_area_mode": "研究区域面积来源",
    "suppressed_count": "抑制数量",
    "symmetric": "对称矩阵",
    "total_component_count": "连通分量总数",
    "total_length": "骨架总长度",
    "total_perimeter": "总边界周长",
    "total_perimeter_px": "总边界周长（px）",
    "total_weight": "梯度总权重",
    "unit": "单位",
    "valid_gradient_pixels": "有效梯度像素数",
    "valid_pixel_count": "有效像素数",
    "valid_sample_count": "有效样本数",
    "vector_area_px": "矢量面积（px²）",
    "z_maximum": "强度最大值",
    "z_minimum": "强度最小值",
}

_VALUE_NAMES = {
    "bright": "亮前景",
    "dark": "暗前景",
    "full_image": "整张图片",
    "luminance": "加权亮度",
    "mask": "掩膜",
    "measurement": "测量对象",
    "roi": "ROI",
    "active_group": "当前类别",
    "all": "当前图片全部计数点",
    "scope": "当前 ROI / 当前视窗",
    "point_bounds": "点集轴对齐包围框",
    "custom": "手工指定",
}


def _display_field_name(name: object) -> str:
    token = str(name)
    return _FIELD_NAMES.get(token, token)


def _display_field_value(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, str):
        return _VALUE_NAMES.get(value.casefold(), value)
    return str(value)


class _CurveCanvas(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._curve: AnalysisCurve | None = None
        self.setMinimumHeight(180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def setCurve(self, curve: AnalysisCurve | None) -> None:
        self._curve = curve
        self.update()

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        plot = QRectF(self.rect()).adjusted(44, 18, -18, -36)
        palette = self.palette()
        painter.fillRect(self.rect(), palette.color(QPalette.ColorRole.Base))
        painter.setPen(QPen(palette.color(QPalette.ColorRole.Mid), 1))
        painter.drawRect(plot)
        curve = self._curve
        valid = (
            ()
            if curve is None
            else tuple(
                (x_value, y_value)
                for x_value, y_value in zip(curve.x, curve.y, strict=True)
                if y_value is not None
            )
        )
        if len(valid) < 2:
            painter.setPen(palette.color(QPalette.ColorRole.PlaceholderText))
            painter.drawText(
                plot.adjusted(12, 12, -12, -12),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                "该结果没有足够的有效数据可绘制。",
            )
            return
        minimum_x = min(point[0] for point in valid)
        maximum_x = max(point[0] for point in valid)
        minimum_y = min(point[1] for point in valid)
        maximum_y = max(point[1] for point in valid)
        range_x = maximum_x - minimum_x or 1.0
        range_y = maximum_y - minimum_y or 1.0
        path = QPainterPath()
        segment_started = False
        for x_value, y_value in zip(curve.x, curve.y, strict=True):
            if y_value is None:
                segment_started = False
                continue
            point_x = plot.left() + ((x_value - minimum_x) / range_x) * plot.width()
            point_y = plot.bottom() - ((y_value - minimum_y) / range_y) * plot.height()
            if segment_started:
                path.lineTo(point_x, point_y)
            else:
                path.moveTo(point_x, point_y)
                segment_started = True
        accent = QColor("#2A9D8F")
        painter.setPen(QPen(accent, 1.8))
        painter.drawPath(path)
        painter.setPen(palette.color(QPalette.ColorRole.Text))
        painter.drawText(
            QRectF(plot.left(), plot.bottom() + 6, plot.width(), 24),
            Qt.AlignmentFlag.AlignCenter,
            f"{curve.name} · X：{curve.x_unit or '无单位'} · Y：{curve.y_unit or '无单位'}",
        )


class AnalysisResultsCenter(QDialog):
    """Browse, locate and export analysis artifacts without mutating them."""

    recalculateRequested = Signal(object)
    convertToMeasurementRequested = Signal(object)
    exportRequested = Signal(object)
    locateRequested = Signal(object)

    def __init__(
        self,
        artifacts: Iterable[AnalysisArtifact] = (),
        *,
        document_names: NameMap = None,
        roi_names: NameMap = None,
        measurement_names: NameMap = None,
        tool_names: NameMap = None,
        asset_root: str | Path | None = None,
        asset_source_paths: Mapping[str, str | Path] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("分析结果中心")
        self.setObjectName("analysisResultsCenter")
        self.setModal(False)
        self.setSizeGripEnabled(True)
        self.setMinimumSize(620, 420)
        self.resize(1040, 700)

        self._artifacts: tuple[AnalysisArtifact, ...] = ()
        self._filtered_artifacts: tuple[AnalysisArtifact, ...] = ()
        self._document_names = dict(document_names or {})
        self._roi_names = dict(roi_names or {})
        self._measurement_names = dict(measurement_names or {})
        self._tool_names = dict(tool_names or {})
        self._asset_root = None if asset_root is None else Path(asset_root)
        self._asset_source_paths = {
            str(key): Path(value)
            for key, value in dict(asset_source_paths or {}).items()
        }
        self._asset_preview_generation = 0
        self._active_preview_task: _AssetPreviewTask | None = None
        self._pending_preview_request: tuple[
            int,
            str,
            Path,
            AnalysisAssetReference,
        ] | None = None
        self._preview_thread_pool = QThreadPool(self)
        self._preview_thread_pool.setMaxThreadCount(1)
        self._preview_thread_pool.setExpiryTimeout(1000)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 10)
        root.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel("分析结果中心", self)
        title.setObjectName("analysisResultsTitle")
        title_font = title.font()
        title_font.setPointSizeF(title_font.pointSizeF() + 2)
        title_font.setBold(True)
        title.setFont(title_font)
        self._selection_status = QLabel("尚未选择分析结果", self)
        self._selection_status.setObjectName("analysisResultStatus")
        self._selection_status.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._selection_status.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        header.addWidget(title)
        header.addWidget(self._selection_status, 1)
        root.addLayout(header)

        self._content_scroll = QScrollArea(self)
        self._content_scroll.setObjectName("analysisResultsScroll")
        self._content_scroll.setProperty("redirectEditorWheel", True)
        self._content_scroll.setWidgetResizable(True)
        self._content_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._content_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._content_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        content = QWidget(self._content_scroll)
        content.setMinimumSize(680, 350)
        content_root = QVBoxLayout(content)
        content_root.setContentsMargins(0, 0, 0, 0)
        content_root.setSpacing(8)

        filter_frame = QFrame(content)
        filter_frame.setObjectName("analysisFilterPanel")
        filter_layout = QGridLayout(filter_frame)
        filter_layout.setContentsMargins(8, 8, 8, 8)
        filter_layout.setHorizontalSpacing(8)
        filter_layout.setVerticalSpacing(6)
        self._document_filter = self._add_filter(
            filter_layout,
            row=0,
            column=0,
            label="文档",
        )
        self._roi_filter = self._add_filter(
            filter_layout,
            row=0,
            column=2,
            label="ROI / 对象",
        )
        self._category_filter = self._add_filter(
            filter_layout,
            row=0,
            column=4,
            label="类别",
        )
        self._tool_filter = self._add_filter(
            filter_layout,
            row=1,
            column=0,
            label="工具",
        )
        self._status_filter = self._add_filter(
            filter_layout,
            row=1,
            column=2,
            label="状态",
        )
        self._status_filter.addItem("全部状态", "")
        self._status_filter.addItem("当前", AnalysisArtifactStatus.CURRENT.value)
        self._status_filter.addItem("已失效", AnalysisArtifactStatus.STALE.value)
        self._count_label = QLabel("0 项结果", filter_frame)
        self._count_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        filter_layout.addWidget(self._count_label, 1, 4, 1, 2)
        for combo in (
            self._document_filter,
            self._roi_filter,
            self._category_filter,
            self._tool_filter,
            self._status_filter,
        ):
            combo.currentIndexChanged.connect(self._refresh_filter)
        content_root.addWidget(filter_frame)

        splitter = QSplitter(Qt.Orientation.Horizontal, content)
        splitter.setObjectName("analysisResultsSplitter")
        splitter.setChildrenCollapsible(False)
        self._artifact_table = QTableWidget(0, 5, splitter)
        self._artifact_table.setObjectName("analysisArtifactTable")
        self._artifact_table.setHorizontalHeaderLabels(
            ("状态", "分析工具", "来源", "摘要", "生成时间")
        )
        self._artifact_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._artifact_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self._artifact_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._artifact_table.setAlternatingRowColors(True)
        self._artifact_table.verticalHeader().setVisible(False)
        artifact_header = self._artifact_table.horizontalHeader()
        artifact_header.setMinimumSectionSize(36)
        for column in (0, 1, 2, 4):
            artifact_header.setSectionResizeMode(
                column,
                QHeaderView.ResizeMode.ResizeToContents,
            )
        artifact_header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self._artifact_table.setMinimumWidth(270)
        self._artifact_table.itemSelectionChanged.connect(self._show_selection)
        self._artifact_table.itemClicked.connect(self._locate_selection)
        self._artifact_table.itemDoubleClicked.connect(self._locate_selection)

        detail = QWidget(splitter)
        detail_layout = QVBoxLayout(detail)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.setSpacing(6)
        self._detail_header = QLabel("选择左侧分析结果以查看详情。", detail)
        self._detail_header.setWordWrap(True)
        detail_layout.addWidget(self._detail_header)
        self._tabs = QTabWidget(detail)
        self._tabs.setDocumentMode(True)
        self._tabs.addTab(self._create_summary_tab(), "分析摘要")
        self._tabs.addTab(self._create_parameters_tab(), "参数与来源")
        self._tabs.addTab(self._create_table_tab(), "详细表格")
        self._tabs.addTab(self._create_curve_tab(), "曲线 / 直方图")
        self._tabs.addTab(self._create_asset_tab(), "标签图 / 资产")
        detail_layout.addWidget(self._tabs, 1)
        splitter.addWidget(self._artifact_table)
        splitter.addWidget(detail)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes((370, 610))
        splitter.setMinimumHeight(280)
        content_root.addWidget(splitter, 1)
        self._content_scroll.setWidget(content)
        root.addWidget(self._content_scroll, 1)

        footer = QHBoxLayout()
        self._locate_button = QPushButton("在画布中定位", self)
        self._recalculate_button = QPushButton("重新计算", self)
        self._convert_button = QPushButton("转换为测量", self)
        self._export_button = QPushButton("导出分析结果…", self)
        self._close_button = QPushButton("关闭", self)
        self._locate_button.clicked.connect(self._locate_selection)
        self._recalculate_button.clicked.connect(self._request_recalculation)
        self._convert_button.clicked.connect(self._request_conversion)
        self._export_button.clicked.connect(self._request_export)
        self._close_button.clicked.connect(self.close)
        footer.addWidget(self._locate_button)
        footer.addWidget(self._recalculate_button)
        footer.addWidget(self._convert_button)
        footer.addStretch(1)
        footer.addWidget(self._export_button)
        footer.addWidget(self._close_button)
        root.addLayout(footer)

        self.setStyleSheet(
            "QFrame#analysisFilterPanel {"
            " background: palette(base);"
            " border: 1px solid palette(mid);"
            " border-radius: 6px;"
            "}"
            "QLabel#analysisResultStatus {"
            " color: palette(placeholder-text);"
            "}"
        )
        self.set_artifacts(artifacts)

    def set_artifacts(
        self,
        artifacts: Iterable[AnalysisArtifact],
        *,
        document_names: NameMap = None,
        roi_names: NameMap = None,
        measurement_names: NameMap = None,
        tool_names: NameMap = None,
    ) -> None:
        frozen = tuple(artifacts)
        if any(not isinstance(item, AnalysisArtifact) for item in frozen):
            raise TypeError("分析结果中心只接受 AnalysisArtifact")
        previous_id = self.current_artifact_id()
        self._artifacts = frozen
        if document_names is not None:
            self._document_names = dict(document_names)
        if roi_names is not None:
            self._roi_names = dict(roi_names)
        if measurement_names is not None:
            self._measurement_names = dict(measurement_names)
        if tool_names is not None:
            self._tool_names = dict(tool_names)
        self._rebuild_filter_choices()
        self._refresh_filter(preferred_artifact_id=previous_id)

    def set_asset_root(self, path: str | Path | None) -> None:
        self._asset_root = None if path is None else Path(path)
        self._show_selection()

    def set_asset_source_paths(
        self,
        paths: Mapping[str, str | Path] | None,
    ) -> None:
        self._asset_source_paths = {
            str(key): Path(value)
            for key, value in dict(paths or {}).items()
        }
        self._show_selection()

    def set_asset_locations(
        self,
        *,
        asset_root: str | Path | None,
        asset_source_paths: Mapping[str, str | Path] | None,
    ) -> None:
        self._asset_root = None if asset_root is None else Path(asset_root)
        self._asset_source_paths = {
            str(key): Path(value)
            for key, value in dict(asset_source_paths or {}).items()
        }
        self._show_selection()

    def filtered_artifacts(self) -> tuple[AnalysisArtifact, ...]:
        return self._filtered_artifacts

    def current_artifact(self) -> AnalysisArtifact | None:
        selected = self._artifact_table.selectionModel().selectedRows()
        if not selected:
            return None
        artifact_id = selected[0].data(Qt.ItemDataRole.UserRole)
        return next(
            (artifact for artifact in self._filtered_artifacts if artifact.id == artifact_id),
            None,
        )

    def current_artifact_id(self) -> str | None:
        artifact = self.current_artifact()
        return None if artifact is None else artifact.id

    def _add_filter(
        self,
        layout: QGridLayout,
        *,
        row: int,
        column: int,
        label: str,
    ) -> NoWheelComboBox:
        label_widget = QLabel(label, layout.parentWidget())
        combo = NoWheelComboBox(layout.parentWidget())
        combo.setMinimumWidth(120)
        combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        layout.addWidget(label_widget, row, column)
        layout.addWidget(combo, row, column + 1)
        return combo

    def _create_summary_tab(self) -> QWidget:
        scroll = QScrollArea(self._tabs)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget(scroll)
        self._summary_form = QFormLayout(content)
        self._summary_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self._summary_labels: dict[str, QLabel] = {}
        for key, title in (
            ("status", "状态"),
            ("document", "来源文档"),
            ("reference", "ROI / 测量对象"),
            ("tool", "分析工具"),
            ("version", "算法版本"),
            ("created_at", "生成时间"),
            ("calibration", "标定"),
            ("reason", "状态说明"),
            ("scalars", "标量结果"),
        ):
            value = QLabel("—", content)
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            value.setWordWrap(True)
            self._summary_labels[key] = value
            self._summary_form.addRow(title, value)
        scroll.setWidget(content)
        return scroll

    def _create_parameters_tab(self) -> QWidget:
        self._parameters_table = QTableWidget(0, 3, self._tabs)
        self._parameters_table.setHorizontalHeaderLabels(("类型", "名称", "值"))
        self._parameters_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._parameters_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._parameters_table.horizontalHeader().setStretchLastSection(True)
        self._parameters_table.verticalHeader().setVisible(False)
        return self._parameters_table

    def _create_table_tab(self) -> QWidget:
        page = QWidget(self._tabs)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 6, 6, 6)
        header = QHBoxLayout()
        header.addWidget(QLabel("结果表", page))
        self._table_selector = NoWheelComboBox(page)
        self._table_selector.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self._table_selector.currentIndexChanged.connect(self._show_current_table)
        header.addWidget(self._table_selector, 1)
        layout.addLayout(header)
        self._detail_table = QTableWidget(0, 0, page)
        self._detail_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._detail_table.setAlternatingRowColors(True)
        self._detail_table.verticalHeader().setVisible(False)
        layout.addWidget(self._detail_table, 1)
        return page

    def _create_curve_tab(self) -> QWidget:
        page = QWidget(self._tabs)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 6, 6, 6)
        header = QHBoxLayout()
        header.addWidget(QLabel("曲线", page))
        self._curve_selector = NoWheelComboBox(page)
        self._curve_selector.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self._curve_selector.currentIndexChanged.connect(self._show_current_curve)
        header.addWidget(self._curve_selector, 1)
        layout.addLayout(header)
        self._curve_canvas = _CurveCanvas(page)
        layout.addWidget(self._curve_canvas, 1)
        return page

    def _create_asset_tab(self) -> QWidget:
        page = QWidget(self._tabs)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 6, 6, 6)
        self._asset_list = QListWidget(page)
        self._asset_list.currentRowChanged.connect(self._show_current_asset)
        layout.addWidget(self._asset_list, 1)
        self._asset_preview = QLabel("没有标签图或分析资产。", page)
        self._asset_preview.setMinimumHeight(120)
        self._asset_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._asset_preview.setWordWrap(True)
        self._asset_preview.setScaledContents(False)
        layout.addWidget(self._asset_preview, 2)
        self._asset_preview_description = QLabel("", page)
        self._asset_preview_description.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        self._asset_preview_description.setWordWrap(True)
        layout.addWidget(self._asset_preview_description)
        return page

    def _rebuild_filter_choices(self) -> None:
        current_values = {
            combo: combo.currentData()
            for combo in (
                self._document_filter,
                self._roi_filter,
                self._category_filter,
                self._tool_filter,
            )
        }
        choices = {
            self._document_filter: {
                artifact.source_document_id: self._document_label(artifact)
                for artifact in self._artifacts
            },
            self._roi_filter: {
                self._reference_filter_key(artifact): self._reference_label(artifact)
                for artifact in self._artifacts
            },
            self._category_filter: {
                category: category
                for artifact in self._artifacts
                if (category := self._category_label(artifact))
            },
            self._tool_filter: {
                artifact.tool_id: self._tool_label(artifact)
                for artifact in self._artifacts
            },
        }
        all_labels = {
            self._document_filter: "全部文档",
            self._roi_filter: "全部 ROI / 对象",
            self._category_filter: "全部类别",
            self._tool_filter: "全部工具",
        }
        for combo, values in choices.items():
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(all_labels[combo], "")
            for key, label in sorted(values.items(), key=lambda item: item[1].casefold()):
                combo.addItem(label, key)
            previous = current_values[combo]
            index = combo.findData(previous)
            combo.setCurrentIndex(max(0, index))
            combo.blockSignals(False)

    def _refresh_filter(
        self,
        _index: int | None = None,
        *,
        preferred_artifact_id: str | None = None,
    ) -> None:
        document_id = str(self._document_filter.currentData() or "")
        reference_key = str(self._roi_filter.currentData() or "")
        category = str(self._category_filter.currentData() or "")
        tool_id = str(self._tool_filter.currentData() or "")
        status = str(self._status_filter.currentData() or "")
        filtered = tuple(
            artifact
            for artifact in self._artifacts
            if (not document_id or artifact.source_document_id == document_id)
            and (not reference_key or self._reference_filter_key(artifact) == reference_key)
            and (not category or self._category_label(artifact) == category)
            and (not tool_id or artifact.tool_id == tool_id)
            and (not status or artifact.status.value == status)
        )
        self._filtered_artifacts = filtered
        self._count_label.setText(f"{len(filtered)} 项结果")
        self._artifact_table.blockSignals(True)
        self._artifact_table.setRowCount(len(filtered))
        selected_row = -1
        for row, artifact in enumerate(filtered):
            values = (
                "当前" if artifact.is_current else "已失效",
                self._tool_label(artifact),
                self._reference_label(artifact),
                self._scalar_summary(artifact),
                self._display_timestamp(artifact.created_at),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, artifact.id)
                if column == 3:
                    item.setToolTip(self._scalar_summary(artifact))
                elif column == 4:
                    item.setToolTip(artifact.created_at)
                if not artifact.is_current:
                    item.setForeground(self.palette().color(QPalette.ColorRole.PlaceholderText))
                    item.setToolTip(artifact.stale_reason or "分析结果已失效")
                self._artifact_table.setItem(row, column, item)
            if artifact.id == preferred_artifact_id:
                selected_row = row
        self._artifact_table.blockSignals(False)
        if selected_row >= 0:
            self._artifact_table.selectRow(selected_row)
        elif filtered:
            self._artifact_table.selectRow(0)
        else:
            self._clear_details()

    def _show_selection(self) -> None:
        artifact = self.current_artifact()
        if artifact is None:
            self._clear_details()
            return
        status_text = "当前" if artifact.is_current else "已失效"
        self._selection_status.setText(
            status_text
            if artifact.is_current
            else f"{status_text} · {artifact.stale_reason or '来源已变化'}"
        )
        self._detail_header.setText(
            f"{self._tool_label(artifact)} · {self._document_label(artifact)}"
        )
        self._summary_labels["status"].setText(status_text)
        self._summary_labels["document"].setText(self._document_label(artifact))
        self._summary_labels["reference"].setText(self._reference_label(artifact))
        self._summary_labels["tool"].setText(self._tool_label(artifact))
        self._summary_labels["version"].setText(artifact.tool_version)
        self._summary_labels["created_at"].setText(artifact.created_at)
        self._summary_labels["calibration"].setText(
            artifact.calibration_signature or "未标定"
        )
        self._summary_labels["reason"].setText(artifact.stale_reason or "来源数据有效")
        self._summary_labels["scalars"].setText(self._scalar_summary(artifact))
        self._populate_parameters(artifact)
        self._populate_tables(artifact)
        self._populate_curves(artifact)
        self._populate_assets(artifact)
        self._set_action_states(artifact)

    def _clear_details(self) -> None:
        self._selection_status.setText("尚未选择分析结果")
        self._detail_header.setText("选择左侧分析结果以查看详情。")
        for label in self._summary_labels.values():
            label.setText("—")
        self._parameters_table.setRowCount(0)
        self._table_selector.clear()
        self._detail_table.setRowCount(0)
        self._detail_table.setColumnCount(0)
        self._curve_selector.clear()
        self._curve_canvas.setCurve(None)
        self._asset_list.clear()
        self._asset_preview.setPixmap(QPixmap())
        self._asset_preview.setText("没有标签图或分析资产。")
        self._asset_preview_description.clear()
        self._set_action_states(None)

    def _populate_parameters(self, artifact: AnalysisArtifact) -> None:
        rows = [
            ("来源", "文档 ID", artifact.source_document_id),
            ("来源", "像素修订", artifact.source_pixel_revision),
            ("来源", "ROI / 对象", self._reference_label(artifact)),
            ("来源", "标定签名", artifact.calibration_signature or "未标定"),
        ]
        rows.extend(
            (
                "参数",
                _display_field_name(name),
                _display_field_value(value),
            )
            for name, value in artifact.parameters.items()
        )
        self._parameters_table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                self._parameters_table.setItem(row, column, QTableWidgetItem(str(value)))
        self._parameters_table.resizeColumnsToContents()

    def _populate_tables(self, artifact: AnalysisArtifact) -> None:
        self._table_selector.blockSignals(True)
        self._table_selector.clear()
        for index, table in enumerate(artifact.tables):
            self._table_selector.addItem(table.name, index)
        self._table_selector.blockSignals(False)
        self._show_current_table()

    def _show_current_table(self, _index: int | None = None) -> None:
        artifact = self.current_artifact()
        index = self._table_selector.currentData()
        if artifact is None or not isinstance(index, int) or not (0 <= index < len(artifact.tables)):
            self._detail_table.setRowCount(0)
            self._detail_table.setColumnCount(0)
            return
        table = artifact.tables[index]
        self._detail_table.setColumnCount(len(table.columns))
        self._detail_table.setHorizontalHeaderLabels(table.columns)
        self._detail_table.setRowCount(len(table.rows))
        for row_index, row in enumerate(table.rows):
            for column_index, value in enumerate(row):
                item = QTableWidgetItem()
                item.setData(Qt.ItemDataRole.DisplayRole, value)
                self._detail_table.setItem(row_index, column_index, item)
        self._detail_table.resizeColumnsToContents()

    def _populate_curves(self, artifact: AnalysisArtifact) -> None:
        self._curve_selector.blockSignals(True)
        self._curve_selector.clear()
        for index, curve in enumerate(artifact.curves):
            self._curve_selector.addItem(curve.name, index)
        self._curve_selector.blockSignals(False)
        self._show_current_curve()

    def _show_current_curve(self, _index: int | None = None) -> None:
        artifact = self.current_artifact()
        index = self._curve_selector.currentData()
        curve = (
            artifact.curves[index]
            if artifact is not None
            and isinstance(index, int)
            and 0 <= index < len(artifact.curves)
            else None
        )
        self._curve_canvas.setCurve(curve)

    def _populate_assets(self, artifact: AnalysisArtifact) -> None:
        self._asset_list.blockSignals(True)
        self._asset_list.clear()
        for index, asset in enumerate(artifact.assets):
            label = {
                AnalysisAssetKind.LABEL_IMAGE: "标签图",
                AnalysisAssetKind.MASK: "掩膜",
                AnalysisAssetKind.GRAPH: "图结构",
                AnalysisAssetKind.TABLE: "大型表格",
                AnalysisAssetKind.CURVE: "大型曲线",
            }.get(asset.kind, "分析资产")
            item = QListWidgetItem(f"{label} · {asset.path}")
            item.setData(Qt.ItemDataRole.UserRole, index)
            item.setToolTip(f"媒体类型：{asset.media_type}\nSHA256：{asset.sha256}")
            self._asset_list.addItem(item)
        self._asset_list.blockSignals(False)
        if artifact.assets:
            self._asset_list.setCurrentRow(0)
        else:
            self._asset_preview.setPixmap(QPixmap())
            self._asset_preview.setText("没有标签图或分析资产。")
            self._asset_preview_description.clear()

    def _show_current_asset(self, row: int) -> None:
        self._asset_preview_generation += 1
        generation = self._asset_preview_generation
        self._cancel_current_asset_preview()
        artifact = self.current_artifact()
        if artifact is None or not (0 <= row < len(artifact.assets)):
            self._asset_preview.setPixmap(QPixmap())
            self._asset_preview.setText("没有标签图或分析资产。")
            self._asset_preview_description.clear()
            return
        asset = artifact.assets[row]
        candidate = self._asset_candidate(asset)
        image = QPixmap(str(candidate)) if candidate is not None and candidate.is_file() else QPixmap()
        if not image.isNull():
            available = self._asset_preview.size().boundedTo(image.size())
            self._asset_preview.setPixmap(
                image.scaled(
                    available,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            self._asset_preview.setText("")
            self._asset_preview_description.setText(
                f"{asset.kind.value} · {asset.path}"
            )
            return
        schema = str(asset.metadata.get("schema", ""))
        if (
            candidate is not None
            and schema in _PREVIEW_SCHEMA_MEMBERS
        ):
            self._asset_preview.setPixmap(QPixmap())
            self._asset_preview.setText("正在安全加载分析预览…")
            self._asset_preview_description.setText(
                f"{schema} · {asset.path}"
            )
            self._queue_asset_preview(
                generation=generation,
                artifact_id=artifact.id,
                candidate=candidate,
                reference=asset,
            )
            return
        self._asset_preview.setPixmap(QPixmap())
        self._asset_preview.setText(
            f"{asset.kind.value}\n{asset.path}\n{asset.media_type}\n"
            "该资产不是可直接预览的图片，或尚未提供项目资产目录。"
        )
        self._asset_preview_description.clear()

    def _cancel_current_asset_preview(self) -> None:
        self._pending_preview_request = None
        active = self._active_preview_task
        if active is None:
            return
        active.cancel()
        if self._preview_thread_pool.tryTake(active):
            self._active_preview_task = None

    def _asset_candidate(
        self,
        reference: AnalysisAssetReference,
    ) -> Path | None:
        mapped = self._asset_source_paths.get(reference.path)
        if mapped is not None:
            return mapped
        if self._asset_root is None:
            return None
        root = self._asset_root.resolve()
        candidate = (root / reference.path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return None
        return candidate

    def _queue_asset_preview(
        self,
        *,
        generation: int,
        artifact_id: str,
        candidate: Path,
        reference: AnalysisAssetReference,
    ) -> None:
        request = (generation, artifact_id, candidate, reference)
        active = self._active_preview_task
        if active is not None:
            active.cancel()
            if self._preview_thread_pool.tryTake(active):
                self._active_preview_task = None
            else:
                self._pending_preview_request = request
                return
        self._pending_preview_request = None
        self._start_asset_preview_task(*request)

    def _start_asset_preview_task(
        self,
        generation: int,
        artifact_id: str,
        candidate: Path,
        reference: AnalysisAssetReference,
    ) -> None:
        if generation != self._asset_preview_generation:
            return
        task = _AssetPreviewTask(
            generation=generation,
            artifact_id=artifact_id,
            candidate=candidate,
            reference=reference,
        )
        task.signals.ready.connect(self._on_asset_preview_ready)
        task.signals.failed.connect(self._on_asset_preview_failed)
        task.signals.finished.connect(self._on_asset_preview_finished)
        self._active_preview_task = task
        self._preview_thread_pool.start(task)

    def _on_asset_preview_ready(self, payload: object) -> None:
        if not isinstance(payload, _AssetPreviewResult):
            return
        artifact = self.current_artifact()
        row = self._asset_list.currentRow()
        if (
            payload.generation != self._asset_preview_generation
            or artifact is None
            or artifact.id != payload.artifact_id
            or not (0 <= row < len(artifact.assets))
            or artifact.assets[row].path != payload.asset_path
        ):
            return
        rgb = np.ascontiguousarray(payload.rgb, dtype=np.uint8)
        height, width, _channels = rgb.shape
        image = QImage(
            rgb.data,
            width,
            height,
            width * 3,
            QImage.Format.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(image)
        available = self._asset_preview.size().boundedTo(pixmap.size())
        self._asset_preview.setPixmap(
            pixmap.scaled(
                available,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self._asset_preview.setText("")
        self._asset_preview_description.setText(payload.description)

    def _on_asset_preview_failed(self, payload: object) -> None:
        if not isinstance(payload, _AssetPreviewFailure):
            return
        artifact = self.current_artifact()
        row = self._asset_list.currentRow()
        if (
            payload.generation != self._asset_preview_generation
            or artifact is None
            or artifact.id != payload.artifact_id
            or not (0 <= row < len(artifact.assets))
            or artifact.assets[row].path != payload.asset_path
        ):
            return
        self._asset_preview.setPixmap(QPixmap())
        self._asset_preview.setText(
            "无法生成该资产的安全预览。\n"
            f"{payload.message}"
        )
        self._asset_preview_description.setText(payload.asset_path)

    def _on_asset_preview_finished(self, generation: int) -> None:
        active = self._active_preview_task
        if active is not None and active.generation == int(generation):
            self._active_preview_task = None
        pending = self._pending_preview_request
        self._pending_preview_request = None
        if pending is not None and pending[0] == self._asset_preview_generation:
            self._start_asset_preview_task(*pending)

    def closeEvent(self, event) -> None:
        self._asset_preview_generation += 1
        self._pending_preview_request = None
        active = self._active_preview_task
        if active is not None:
            active.cancel()
            self._preview_thread_pool.tryTake(active)
            self._active_preview_task = None
        self._preview_thread_pool.clear()
        self._preview_thread_pool.waitForDone(1000)
        super().closeEvent(event)

    def _set_action_states(self, artifact: AnalysisArtifact | None) -> None:
        available = artifact is not None
        self._locate_button.setEnabled(available)
        self._recalculate_button.setEnabled(available)
        self._convert_button.setEnabled(
            artifact is not None
            and artifact.is_current
            and self._is_convertible(artifact)
        )
        self._export_button.setEnabled(bool(self._filtered_artifacts))

    def _locate_selection(self, *_args) -> None:
        artifact = self.current_artifact()
        if artifact is None:
            return
        reference = artifact.source_reference
        self.locateRequested.emit(
            AnalysisLocateRequest(
                artifact_id=artifact.id,
                document_id=artifact.source_document_id,
                object_kind=None if reference is None else reference.kind.value,
                object_id=None if reference is None else reference.object_id,
            )
        )

    def _request_recalculation(self) -> None:
        artifact = self.current_artifact()
        if artifact is not None:
            self.recalculateRequested.emit(AnalysisActionRequest((artifact.id,)))

    def _request_conversion(self) -> None:
        artifact = self.current_artifact()
        if artifact is not None and artifact.is_current and self._is_convertible(artifact):
            self.convertToMeasurementRequested.emit(
                AnalysisActionRequest((artifact.id,))
            )

    def _request_export(self) -> None:
        artifact_ids = tuple(artifact.id for artifact in self._filtered_artifacts)
        if not artifact_ids:
            return
        table_name = (
            self._table_selector.currentText().strip()
            if len(artifact_ids) == 1 and self._table_selector.count()
            else None
        )
        self.exportRequested.emit(
            AnalysisExportRequest(
                artifact_ids=artifact_ids,
                selected_table_name=table_name or None,
            )
        )

    def _document_label(self, artifact: AnalysisArtifact) -> str:
        return self._document_names.get(
            artifact.source_document_id,
            artifact.source_document_id,
        )

    def _reference_label(self, artifact: AnalysisArtifact) -> str:
        reference = artifact.source_reference
        if reference is None:
            return "整张图片"
        if reference.kind is AnalysisObjectKind.ROI:
            name = self._roi_names.get(reference.object_id, reference.object_id)
            return f"ROI：{name}"
        name = self._measurement_names.get(reference.object_id, reference.object_id)
        return f"测量对象：{name}"

    @staticmethod
    def _reference_filter_key(artifact: AnalysisArtifact) -> str:
        reference = artifact.source_reference
        return (
            "__whole_image__"
            if reference is None
            else f"{reference.kind.value}:{reference.object_id}"
        )

    @staticmethod
    def _category_label(artifact: AnalysisArtifact) -> str:
        parameters = artifact.parameters
        for key in ("category_label", "category", "类别"):
            value = parameters.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _tool_label(self, artifact: AnalysisArtifact) -> str:
        explicit = self._tool_names.get(artifact.tool_id)
        if explicit:
            return explicit
        value = artifact.parameters.get("tool_name")
        if isinstance(value, str) and value.strip():
            return value.strip()
        return _TOOL_NAMES.get(artifact.tool_id, artifact.tool_id)

    @staticmethod
    def _scalar_summary(artifact: AnalysisArtifact) -> str:
        scalars = artifact.scalars
        if not scalars:
            return "无标量摘要"
        return "；".join(
            f"{_display_field_name(name)}={_display_field_value(value)}"
            for name, value in tuple(scalars.items())[:4]
        )

    @staticmethod
    def _display_timestamp(value: str) -> str:
        return str(value).replace("T", " ")[:16]

    @staticmethod
    def _is_convertible(artifact: AnalysisArtifact) -> bool:
        token = artifact.tool_id.casefold()
        explicit = artifact.parameters.get("convertible")
        return explicit is True or "particle" in token or any(
            marker in token for marker in ("maxima", "extrema")
        )


__all__ = [
    "AnalysisActionRequest",
    "AnalysisExportRequest",
    "AnalysisLocateRequest",
    "AnalysisResultsCenter",
]
