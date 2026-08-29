from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import math
from pathlib import Path
import random
import re
import shutil
import tempfile
from typing import Callable, Iterable

import cv2
import numpy as np
from PIL import Image
import tifffile


class DatasetExportFormat(str, Enum):
    COCO_INSTANCE = "coco_instance"
    YOLO_DETECTION = "yolo_detection"
    YOLO_SEGMENTATION = "yolo_segmentation"
    SEMANTIC_LABELS = "semantic_labels"
    INSTANCE_LABELS = "instance_labels"

    @property
    def label(self) -> str:
        return {
            self.COCO_INSTANCE: "COCO 实例分割（RLE）",
            self.YOLO_DETECTION: "YOLO 检测",
            self.YOLO_SEGMENTATION: "YOLO 分割",
            self.SEMANTIC_LABELS: "语义分割标签图",
            self.INSTANCE_LABELS: "实例分割标签图",
        }[self]


class DatasetIssueSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(slots=True)
class DatasetInstance:
    instance_id: str
    category_name: str
    rings_px: list[list[tuple[float, float]]]
    source_object_id: str = ""
    confidence: float = 1.0
    truncated: bool = False
    source_verified: bool = True


@dataclass(slots=True)
class DatasetSample:
    sample_id: str
    source_group_id: str
    source_name: str
    image: np.ndarray
    instances: list[DatasetInstance] = field(default_factory=list)
    valid_coverage: np.ndarray | None = None
    focus_index: int | None = None
    origin_px: tuple[int, int] = (0, 0)
    annotation_complete: bool = False
    source_metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        array = np.asarray(self.image)
        if array.ndim not in {2, 3} or array.shape[0] <= 0 or array.shape[1] <= 0:
            raise ValueError(f"训练样本 {self.sample_id!r} 的图像尺寸无效。")
        if array.ndim == 3 and array.shape[2] not in {3, 4}:
            raise ValueError(f"训练样本 {self.sample_id!r} 的通道数不受支持。")
        self.image = np.ascontiguousarray(array)
        if self.valid_coverage is not None:
            coverage = np.asarray(self.valid_coverage, dtype=bool)
            if coverage.shape != array.shape[:2]:
                raise ValueError(f"训练样本 {self.sample_id!r} 的有效覆盖尺寸不匹配。")
            self.valid_coverage = np.ascontiguousarray(coverage)

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])


@dataclass(frozen=True, slots=True)
class DatasetPreflightIssue:
    severity: DatasetIssueSeverity
    code: str
    message: str
    sample_id: str = ""
    instance_id: str = ""
    count: int = 1


@dataclass(slots=True)
class DatasetExportRequest:
    output_directory: Path
    samples: list[DatasetSample]
    formats: tuple[DatasetExportFormat, ...] = (
        DatasetExportFormat.COCO_INSTANCE,
    )
    category_mapping: dict[str, str | None] = field(default_factory=dict)
    split_train_validation: bool = True
    validation_fraction: float = 0.2
    random_seed: int = 20260829
    yolo_complex_policy: str = "skip"
    convert_high_bit_to_uint8: bool = False
    source_issues: list[DatasetPreflightIssue] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class DatasetExportResult:
    output_directory: Path
    sample_count: int
    instance_count: int
    skipped_instance_count: int
    category_names: tuple[str, ...]
    formats: tuple[str, ...]
    issues: tuple[DatasetPreflightIssue, ...]


@dataclass(slots=True)
class _PreparedInstance:
    source: DatasetInstance
    category_name: str
    category_id: int
    yolo_class_id: int
    mask: np.ndarray
    bbox: tuple[int, int, int, int]


@dataclass(slots=True)
class _PreparedSample:
    source: DatasetSample
    file_stem: str
    split: str
    instances: list[_PreparedInstance]


def _normalized_category_key(value: str) -> str:
    return " ".join(str(value or "").strip().split()).casefold()


def _normalized_category_name(value: object) -> str | None:
    if value is None:
        return None
    normalized = " ".join(str(value).strip().split())
    return normalized or None


def _safe_stem(value: str, *, fallback: str) -> str:
    token = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value or "").strip())
    token = token.strip("._-")[:100]
    return token or fallback


def _rings_mask(
    rings: Iterable[Iterable[tuple[float, float]]],
    *,
    width: int,
    height: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for ring in rings:
        points = list(ring)
        if len(points) < 3:
            continue
        coordinates: list[list[int]] = []
        try:
            for x, y in points:
                x_value = float(x)
                y_value = float(y)
                if not (math.isfinite(x_value) and math.isfinite(y_value)):
                    raise ValueError
                coordinates.append(
                    [
                        int(np.clip(round(x_value), -(2**30), 2**30)),
                        int(np.clip(round(y_value), -(2**30), 2**30)),
                    ]
                )
        except (TypeError, ValueError, OverflowError):
            continue
        contour = np.asarray(coordinates, dtype=np.int32)
        ring_mask = np.zeros_like(mask)
        cv2.fillPoly(ring_mask, [contour], 1)
        mask ^= ring_mask
    return mask.astype(bool)


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    left = int(xs.min())
    top = int(ys.min())
    return left, top, int(xs.max()) - left + 1, int(ys.max()) - top + 1


def _mask_yolo_contours(mask: np.ndarray) -> tuple[list[np.ndarray], bool]:
    contours, hierarchy = cv2.findContours(
        np.asarray(mask, dtype=np.uint8),
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if hierarchy is None:
        return [], False
    entries = hierarchy[0]
    outer = [
        contour.reshape(-1, 2)
        for index, contour in enumerate(contours)
        if int(entries[index][3]) < 0 and len(contour) >= 3
    ]
    has_holes = any(int(entry[3]) >= 0 for entry in entries)
    return outer, has_holes or len(outer) != 1


def _uncompressed_coco_rle(mask: np.ndarray) -> dict[str, object]:
    flattened = np.asarray(mask, dtype=np.uint8).reshape(-1, order="F")
    counts: list[int] = []
    previous = 0
    run = 0
    for value in flattened:
        current = int(value)
        if current == previous:
            run += 1
            continue
        counts.append(run)
        run = 1
        previous = current
    counts.append(run)
    return {
        "size": [int(mask.shape[0]), int(mask.shape[1])],
        "counts": counts,
    }


def _image_suffix(array: np.ndarray, *, convert_high_bit: bool) -> str:
    if array.dtype == np.uint8:
        return ".png"
    return ".png" if convert_high_bit else ".tif"


def _uint8_image(array: np.ndarray) -> np.ndarray:
    source = np.asarray(array)
    if source.dtype == np.uint8:
        return source
    values = source.astype(np.float64, copy=False)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(source.shape, dtype=np.uint8)
    low = float(values[finite].min())
    high = float(values[finite].max())
    if high <= low:
        result = np.zeros(source.shape, dtype=np.uint8)
        result[finite] = 128
        return result
    normalized = np.nan_to_num((values - low) / (high - low), nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(np.rint(normalized * 255.0), 0, 255).astype(np.uint8)


def _write_source_image(path: Path, array: np.ndarray, *, convert_high_bit: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.casefold() in {".tif", ".tiff"}:
        tifffile.imwrite(path, np.asarray(array), compression="deflate")
        return
    output = _uint8_image(array) if convert_high_bit else np.asarray(array)
    Image.fromarray(output).save(path, format="PNG", compress_level=6)


class DatasetExportService:
    """Model-independent export of confirmed area annotations."""

    def preflight(self, request: DatasetExportRequest) -> list[DatasetPreflightIssue]:
        issues: list[DatasetPreflightIssue] = list(request.source_issues)
        if not request.samples:
            issues.append(
                DatasetPreflightIssue(
                    DatasetIssueSeverity.ERROR,
                    "no_samples",
                    "没有可导出的图片或数字切片局部样本。",
                )
            )
            return issues
        if not request.formats:
            issues.append(
                DatasetPreflightIssue(
                    DatasetIssueSeverity.ERROR,
                    "no_formats",
                    "至少选择一种训练数据格式。",
                )
            )
        if request.output_directory.exists():
            issues.append(
                DatasetPreflightIssue(
                    DatasetIssueSeverity.ERROR,
                    "output_exists",
                    "目标目录已经存在；为避免覆盖训练数据，请选择一个新目录。",
                )
            )

        normalized_mapping = {
            _normalized_category_key(source): _normalized_category_name(target)
            for source, target in request.category_mapping.items()
        }
        mapped_names: dict[str, str] = {}
        for sample in request.samples:
            if not sample.annotation_complete:
                issues.append(
                    DatasetPreflightIssue(
                        DatasetIssueSeverity.WARNING,
                        "annotation_completeness_unconfirmed",
                        "尚未确认该样本是否已完整标注所有目标；部分标注会让未标对象被训练为背景。",
                        sample_id=sample.sample_id,
                    )
                )
            if not sample.instances:
                issues.append(
                    DatasetPreflightIssue(
                        DatasetIssueSeverity.WARNING,
                        "empty_sample_unconfirmed",
                        "样本没有面积标注；请确认它确实是无目标负样本。",
                        sample_id=sample.sample_id,
                    )
                )
            if sample.valid_coverage is not None and not bool(sample.valid_coverage.all()):
                issues.append(
                    DatasetPreflightIssue(
                        DatasetIssueSeverity.WARNING,
                        "partial_source_coverage",
                        "数字切片局部样本包含缺失图块；接触缺失区域的对象将被标记为截断。",
                        sample_id=sample.sample_id,
                    )
                )

            prepared_masks: list[tuple[str, np.ndarray]] = []
            for instance in sample.instances:
                source_key = _normalized_category_key(instance.category_name)
                target = normalized_mapping.get(source_key)
                if target is None:
                    issues.append(
                        DatasetPreflightIssue(
                            DatasetIssueSeverity.WARNING,
                            "category_excluded",
                            f"类别“{instance.category_name or '未分类'}”尚未映射，将跳过相关实例。",
                            sample_id=sample.sample_id,
                            instance_id=instance.instance_id,
                        )
                    )
                    continue
                target_key = _normalized_category_key(target)
                previous = mapped_names.setdefault(target_key, target)
                if previous != target:
                    issues.append(
                        DatasetPreflightIssue(
                            DatasetIssueSeverity.WARNING,
                            "category_case_collision",
                            f"导出类别“{previous}”与“{target}”仅大小写不同，将合并为同一类别。",
                        )
                    )
                coordinates = [point for ring in instance.rings_px for point in ring]
                coordinates_outside = False
                try:
                    coordinates_outside = any(
                        not (
                            math.isfinite(float(x))
                            and math.isfinite(float(y))
                            and 0.0 <= float(x) <= sample.width - 1
                            and 0.0 <= float(y) <= sample.height - 1
                        )
                        for x, y in coordinates
                    )
                except (TypeError, ValueError, OverflowError):
                    coordinates_outside = True
                if coordinates_outside:
                    issues.append(
                        DatasetPreflightIssue(
                            DatasetIssueSeverity.WARNING,
                            "geometry_out_of_bounds",
                            "面积对象部分超出源图边界；导出时会按源图范围裁切。",
                            sample_id=sample.sample_id,
                            instance_id=instance.instance_id,
                        )
                    )
                mask = _rings_mask(instance.rings_px, width=sample.width, height=sample.height)
                if sample.valid_coverage is not None:
                    outside = bool(np.any(mask & ~sample.valid_coverage))
                    mask &= sample.valid_coverage
                    if outside:
                        issues.append(
                            DatasetPreflightIssue(
                                DatasetIssueSeverity.WARNING,
                                "instance_hits_missing_coverage",
                                "实例接触数字切片缺失区域，导出轮廓会被裁切。",
                                sample_id=sample.sample_id,
                                instance_id=instance.instance_id,
                            )
                        )
                if not mask.any():
                    issues.append(
                        DatasetPreflightIssue(
                            DatasetIssueSeverity.WARNING,
                            "invalid_geometry",
                            "面积对象几何为空、越界或无法栅格化，将跳过。",
                            sample_id=sample.sample_id,
                            instance_id=instance.instance_id,
                        )
                    )
                    continue
                if instance.truncated or not instance.source_verified:
                    issues.append(
                        DatasetPreflightIssue(
                            DatasetIssueSeverity.WARNING,
                            "unverified_or_truncated",
                            "对象来源焦层未核实或触及识别边界；请确认是否适合用于训练。",
                            sample_id=sample.sample_id,
                            instance_id=instance.instance_id,
                        )
                    )
                if (
                    DatasetExportFormat.YOLO_SEGMENTATION in request.formats
                ):
                    _outer_contours, yolo_topology_complex = _mask_yolo_contours(mask)
                    if yolo_topology_complex:
                        issues.append(
                            DatasetPreflightIssue(
                                DatasetIssueSeverity.WARNING,
                                "yolo_topology_loss",
                                "YOLO 分割不能无损表示孔洞或多连通区域；将按当前策略跳过或有损转换。",
                                sample_id=sample.sample_id,
                                instance_id=instance.instance_id,
                            )
                        )
                prepared_masks.append((instance.instance_id, mask))

            if len(prepared_masks) <= 500:
                occupied = np.zeros((sample.height, sample.width), dtype=np.uint16)
                for _instance_id, mask in prepared_masks:
                    occupied += mask.astype(np.uint16)
                overlap_pixels = int(np.count_nonzero(occupied > 1))
                if overlap_pixels:
                    issues.append(
                        DatasetPreflightIssue(
                            DatasetIssueSeverity.WARNING,
                            "overlapping_instances",
                            f"存在 {overlap_pixels} 个重叠像素；COCO 可保留重叠，标签图会写入忽略值。",
                            sample_id=sample.sample_id,
                            count=overlap_pixels,
                        )
                    )
            elif prepared_masks:
                issues.append(
                    DatasetPreflightIssue(
                        DatasetIssueSeverity.INFO,
                        "overlap_check_simplified",
                        "该样本实例超过 500 个；为控制内存占用，导出前不逐像素检查实例重叠。",
                        sample_id=sample.sample_id,
                        count=len(prepared_masks),
                    )
                )
        if request.split_train_validation:
            groups = {sample.source_group_id for sample in request.samples}
            if len(groups) < 2:
                issues.append(
                    DatasetPreflightIssue(
                        DatasetIssueSeverity.WARNING,
                        "single_source_no_split",
                        "只有一个独立来源，不能在不泄漏相邻图块/焦层的前提下划分训练集和验证集。",
                    )
                )
        return self._collapse_issues(issues)

    @staticmethod
    def _collapse_issues(issues: list[DatasetPreflightIssue]) -> list[DatasetPreflightIssue]:
        collapsed: dict[tuple[object, ...], DatasetPreflightIssue] = {}
        for issue in issues:
            key = (issue.severity, issue.code, issue.message, issue.sample_id)
            previous = collapsed.get(key)
            if previous is None:
                collapsed[key] = issue
            else:
                collapsed[key] = DatasetPreflightIssue(
                    severity=previous.severity,
                    code=previous.code,
                    message=previous.message,
                    sample_id=previous.sample_id,
                    count=previous.count + issue.count,
                )
        return list(collapsed.values())

    def export(
        self,
        request: DatasetExportRequest,
        *,
        cancellation_requested: Callable[[], bool] | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> DatasetExportResult:
        issues = self.preflight(request)
        errors = [issue for issue in issues if issue.severity is DatasetIssueSeverity.ERROR]
        if errors:
            raise ValueError("；".join(issue.message for issue in errors))
        target = Path(request.output_directory).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
        try:
            prepared, categories, skipped = self._prepare(request)
            total_steps = max(1, len(prepared) + len(request.formats))
            for index, sample in enumerate(prepared, start=1):
                self._raise_if_cancelled(cancellation_requested)
                suffix = _image_suffix(
                    sample.source.image,
                    convert_high_bit=request.convert_high_bit_to_uint8,
                )
                image_path = staging / "images" / sample.split / f"{sample.file_stem}{suffix}"
                _write_source_image(
                    image_path,
                    sample.source.image,
                    convert_high_bit=request.convert_high_bit_to_uint8,
                )
                if progress_callback is not None:
                    progress_callback(index, total_steps, f"写入源图：{sample.source.source_name}")

            completed = len(prepared)
            for format_value in request.formats:
                self._raise_if_cancelled(cancellation_requested)
                if format_value is DatasetExportFormat.COCO_INSTANCE:
                    self._write_coco(staging, prepared, categories, request)
                elif format_value is DatasetExportFormat.YOLO_DETECTION:
                    self._write_yolo_detection(staging, prepared, categories, request)
                elif format_value is DatasetExportFormat.YOLO_SEGMENTATION:
                    self._write_yolo_segmentation(staging, prepared, categories, request)
                elif format_value is DatasetExportFormat.SEMANTIC_LABELS:
                    self._write_semantic_labels(staging, prepared, categories)
                elif format_value is DatasetExportFormat.INSTANCE_LABELS:
                    self._write_instance_labels(staging, prepared, categories)
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, total_steps, f"生成{format_value.label}")

            report_payload = {
                "schema_version": 1,
                "sample_count": len(prepared),
                "instance_count": sum(len(item.instances) for item in prepared),
                "skipped_instance_count": skipped,
                "formats": [item.value for item in request.formats],
                "categories": [
                    {"id": index + 1, "name": name}
                    for index, name in enumerate(categories)
                ],
                "split": {
                    "enabled": bool(request.split_train_validation),
                    "validation_fraction": float(request.validation_fraction),
                    "seed": int(request.random_seed),
                    "grouped_by_independent_source": True,
                },
                "high_bit_conversion": (
                    "per-sample finite min-max to uint8"
                    if request.convert_high_bit_to_uint8
                    else "preserved"
                ),
                "issues": [
                    {
                        "severity": issue.severity.value,
                        "code": issue.code,
                        "message": issue.message,
                        "sample_id": issue.sample_id,
                        "count": issue.count,
                    }
                    for issue in issues
                ],
                "samples": [
                    {
                        "sample_id": item.source.sample_id,
                        "source_group_id": item.source.source_group_id,
                        "source_name": item.source.source_name,
                        "focus_index": item.source.focus_index,
                        "origin_px": list(item.source.origin_px),
                        "split": item.split,
                        "source_metadata": item.source.source_metadata,
                    }
                    for item in prepared
                ],
            }
            (staging / "export_report.json").write_text(
                json.dumps(report_payload, ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )
            self._raise_if_cancelled(cancellation_requested)
            staging.rename(target)
            return DatasetExportResult(
                output_directory=target,
                sample_count=len(prepared),
                instance_count=sum(len(item.instances) for item in prepared),
                skipped_instance_count=skipped,
                category_names=tuple(categories),
                formats=tuple(item.value for item in request.formats),
                issues=tuple(issues),
            )
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    @staticmethod
    def _raise_if_cancelled(callback: Callable[[], bool] | None) -> None:
        if callback is not None and callback():
            raise RuntimeError("训练数据导出已取消。")

    def _prepare(
        self,
        request: DatasetExportRequest,
    ) -> tuple[list[_PreparedSample], list[str], int]:
        mapping = {
            _normalized_category_key(source): _normalized_category_name(target)
            for source, target in request.category_mapping.items()
        }
        names_by_key: dict[str, str] = {}
        for sample in request.samples:
            for instance in sample.instances:
                target = mapping.get(_normalized_category_key(instance.category_name))
                if target:
                    names_by_key.setdefault(_normalized_category_key(target), target)
        categories = [names_by_key[key] for key in sorted(names_by_key)]
        category_ids = {
            _normalized_category_key(name): index + 1
            for index, name in enumerate(categories)
        }
        splits = self._split_by_source_group(request)
        prepared: list[_PreparedSample] = []
        used_stems: set[str] = set()
        skipped = 0
        for sample_index, sample in enumerate(request.samples, start=1):
            base = _safe_stem(sample.sample_id or sample.source_name, fallback=f"sample_{sample_index:05d}")
            stem = base
            suffix_index = 2
            while stem.casefold() in used_stems:
                stem = f"{base}_{suffix_index}"
                suffix_index += 1
            used_stems.add(stem.casefold())
            prepared_instances: list[_PreparedInstance] = []
            for instance in sample.instances:
                target = mapping.get(_normalized_category_key(instance.category_name))
                if not target:
                    skipped += 1
                    continue
                category_id = category_ids.get(_normalized_category_key(target))
                if category_id is None:
                    skipped += 1
                    continue
                mask = _rings_mask(instance.rings_px, width=sample.width, height=sample.height)
                if sample.valid_coverage is not None:
                    mask &= sample.valid_coverage
                bbox = _mask_bbox(mask)
                if bbox is None:
                    skipped += 1
                    continue
                prepared_instances.append(
                    _PreparedInstance(
                        source=instance,
                        category_name=target,
                        category_id=category_id,
                        yolo_class_id=category_id - 1,
                        mask=mask,
                        bbox=bbox,
                    )
                )
            prepared.append(
                _PreparedSample(
                    source=sample,
                    file_stem=stem,
                    split=splits.get(sample.source_group_id, "all"),
                    instances=prepared_instances,
                )
            )
        return prepared, categories, skipped

    @staticmethod
    def _split_by_source_group(request: DatasetExportRequest) -> dict[str, str]:
        groups = sorted({sample.source_group_id for sample in request.samples})
        if not request.split_train_validation or len(groups) < 2:
            return {group: "all" for group in groups}
        rng = random.Random(int(request.random_seed))
        rng.shuffle(groups)
        fraction = max(0.05, min(0.5, float(request.validation_fraction)))
        validation_count = max(1, min(len(groups) - 1, int(round(len(groups) * fraction))))
        validation = set(groups[:validation_count])
        return {group: ("val" if group in validation else "train") for group in groups}

    @staticmethod
    def _image_relative_path(sample: _PreparedSample, request: DatasetExportRequest) -> str:
        suffix = _image_suffix(sample.source.image, convert_high_bit=request.convert_high_bit_to_uint8)
        return f"images/{sample.split}/{sample.file_stem}{suffix}"

    @staticmethod
    def _publish_format_image(
        root: Path,
        folder: str,
        sample: _PreparedSample,
        request: DatasetExportRequest,
    ) -> Path:
        source = root / DatasetExportService._image_relative_path(sample, request)
        target = root / folder / "images" / sample.split / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            return target
        try:
            target.hardlink_to(source)
        except OSError:
            shutil.copy2(source, target)
        return target

    def _write_coco(
        self,
        root: Path,
        samples: list[_PreparedSample],
        categories: list[str],
        request: DatasetExportRequest,
    ) -> None:
        for sample in samples:
            self._publish_format_image(root, "coco", sample, request)

        def payload_for(selected: list[_PreparedSample]) -> dict[str, object]:
            images: list[dict[str, object]] = []
            annotations: list[dict[str, object]] = []
            annotation_id = 1
            for image_id, sample in enumerate(selected, start=1):
                suffix = _image_suffix(
                    sample.source.image,
                    convert_high_bit=request.convert_high_bit_to_uint8,
                )
                images.append(
                    {
                        "id": image_id,
                        "file_name": f"images/{sample.split}/{sample.file_stem}{suffix}",
                        "width": sample.source.width,
                        "height": sample.source.height,
                        "split": sample.split,
                        "focus_index": sample.source.focus_index,
                    }
                )
                for instance in sample.instances:
                    left, top, width, height = instance.bbox
                    annotations.append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": instance.category_id,
                            "segmentation": _uncompressed_coco_rle(instance.mask),
                            "area": int(np.count_nonzero(instance.mask)),
                            "bbox": [left, top, width, height],
                            "iscrowd": 0,
                            "source_object_id": instance.source.source_object_id,
                        }
                    )
                    annotation_id += 1
            return {
                "info": {"description": "FDM confirmed area annotations", "version": "1.0"},
                "licenses": [],
                "images": images,
                "annotations": annotations,
                "categories": [
                    {"id": index + 1, "name": name, "supercategory": "fiber"}
                    for index, name in enumerate(categories)
                ],
            }

        target_root = root / "coco"
        target_root.mkdir(parents=True, exist_ok=True)
        combined = payload_for(samples)
        (target_root / "annotations.json").write_text(
            json.dumps(combined, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        annotations_root = target_root / "annotations"
        annotations_root.mkdir(parents=True, exist_ok=True)
        for split in sorted({sample.split for sample in samples}):
            payload = payload_for([sample for sample in samples if sample.split == split])
            (annotations_root / f"instances_{split}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )

    @staticmethod
    def _write_yolo_metadata(
        root: Path,
        folder: str,
        categories: list[str],
        samples: list[_PreparedSample],
    ) -> None:
        target = root / folder / "dataset.yaml"
        target.parent.mkdir(parents=True, exist_ok=True)
        splits = {sample.split for sample in samples}
        lines = ["path: ."]
        if splits == {"all"}:
            lines.append("train: images/all")
        else:
            if "train" in splits:
                lines.append("train: images/train")
            if "val" in splits:
                lines.append("val: images/val")
        if categories:
            lines.append("names:")
            lines.extend(
                f"  {index}: {json.dumps(name, ensure_ascii=False, allow_nan=False)}"
                for index, name in enumerate(categories)
            )
        else:
            lines.append("names: {}")
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_yolo_detection(
        self,
        root: Path,
        samples: list[_PreparedSample],
        categories: list[str],
        request: DatasetExportRequest,
    ) -> None:
        folder = "yolo_detection"
        self._write_yolo_metadata(root, folder, categories, samples)
        for sample in samples:
            self._publish_format_image(root, folder, sample, request)
            lines: list[str] = []
            for instance in sample.instances:
                left, top, width, height = instance.bbox
                center_x = (left + width / 2.0) / sample.source.width
                center_y = (top + height / 2.0) / sample.source.height
                lines.append(
                    f"{instance.yolo_class_id} {center_x:.8f} {center_y:.8f} "
                    f"{width / sample.source.width:.8f} {height / sample.source.height:.8f}"
                )
            target = root / folder / "labels" / sample.split / f"{sample.file_stem}.txt"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def _write_yolo_segmentation(
        self,
        root: Path,
        samples: list[_PreparedSample],
        categories: list[str],
        request: DatasetExportRequest,
    ) -> None:
        folder = "yolo_segmentation"
        self._write_yolo_metadata(root, folder, categories, samples)
        for sample in samples:
            self._publish_format_image(root, folder, sample, request)
            lines: list[str] = []
            for instance in sample.instances:
                contours, complex_topology = _mask_yolo_contours(instance.mask)
                if not contours:
                    continue
                if complex_topology and request.yolo_complex_policy != "lossy_largest_outer":
                    continue
                ring = max(
                    contours,
                    key=lambda item: abs(cv2.contourArea(np.asarray(item, dtype=np.float32))),
                )
                coordinates: list[str] = []
                for x, y in ring:
                    normalized_x = float(np.clip(float(x) / sample.source.width, 0.0, 1.0))
                    normalized_y = float(np.clip(float(y) / sample.source.height, 0.0, 1.0))
                    coordinates.extend(
                        (
                            f"{normalized_x:.8f}",
                            f"{normalized_y:.8f}",
                        )
                    )
                lines.append(f"{instance.yolo_class_id} " + " ".join(coordinates))
            target = root / folder / "labels" / sample.split / f"{sample.file_stem}.txt"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    @staticmethod
    def _write_semantic_labels(
        root: Path,
        samples: list[_PreparedSample],
        categories: list[str],
    ) -> None:
        mapping = {
            "background": 0,
            "ignore": 65535,
            "categories": {name: index + 1 for index, name in enumerate(categories)},
        }
        target_root = root / "semantic_labels"
        target_root.mkdir(parents=True, exist_ok=True)
        (target_root / "classes.json").write_text(
            json.dumps(mapping, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        for sample in samples:
            labels = np.zeros((sample.source.height, sample.source.width), dtype=np.uint16)
            occupied = np.zeros_like(labels, dtype=bool)
            overlap = np.zeros_like(labels, dtype=bool)
            for instance in sample.instances:
                overlap |= occupied & instance.mask
                labels[instance.mask & ~occupied] = instance.category_id
                occupied |= instance.mask
            labels[overlap] = np.uint16(65535)
            if sample.source.valid_coverage is not None:
                labels[~sample.source.valid_coverage] = np.uint16(65535)
            target = target_root / "labels" / sample.split / f"{sample.file_stem}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(labels).save(target, format="PNG")

    @staticmethod
    def _write_instance_labels(
        root: Path,
        samples: list[_PreparedSample],
        categories: list[str],
    ) -> None:
        target_root = root / "instance_labels"
        target_root.mkdir(parents=True, exist_ok=True)
        (target_root / "classes.json").write_text(
            json.dumps(
                {"categories": {name: index + 1 for index, name in enumerate(categories)}},
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        for sample in samples:
            labels = np.zeros((sample.source.height, sample.source.width), dtype=np.uint32)
            occupied = np.zeros_like(labels, dtype=bool)
            overlap = np.zeros_like(labels, dtype=bool)
            instances_payload: list[dict[str, object]] = []
            for local_id, instance in enumerate(sample.instances, start=1):
                overlap |= occupied & instance.mask
                labels[instance.mask & ~occupied] = np.uint32(local_id)
                occupied |= instance.mask
                instances_payload.append(
                    {
                        "instance_id": local_id,
                        "category_id": instance.category_id,
                        "category_name": instance.category_name,
                        "source_object_id": instance.source.source_object_id,
                    }
                )
            labels[overlap] = np.uint32(0xFFFFFFFF)
            if sample.source.valid_coverage is not None:
                labels[~sample.source.valid_coverage] = np.uint32(0xFFFFFFFF)
            labels_path = target_root / "labels" / sample.split / f"{sample.file_stem}.tif"
            labels_path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(labels_path, labels, compression="deflate")
            mapping_path = target_root / "mapping" / sample.split / f"{sample.file_stem}.json"
            mapping_path.parent.mkdir(parents=True, exist_ok=True)
            mapping_path.write_text(
                json.dumps(
                    {"ignore_value": 0xFFFFFFFF, "instances": instances_payload},
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
