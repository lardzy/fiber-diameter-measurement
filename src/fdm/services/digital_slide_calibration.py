from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Callable

from PySide6.QtGui import QImage

from fdm.services.digital_slide_cache import DigitalSlideSessionCache
from fdm.services.digital_slide_store import (
    DigitalSlideManifest,
    DigitalSlideStore,
    DigitalSlideTileDescriptor,
)
from fdm.services.preview_analysis import estimate_tile_translation


CALIBRATION_AXIS_X = "x"
CALIBRATION_AXIS_Y = "y"


@dataclass(frozen=True, slots=True)
class DigitalSlideCalibrationPair:
    reference: DigitalSlideTileDescriptor
    candidate: DigitalSlideTileDescriptor
    axis: str

    @property
    def nominal_dx(self) -> int:
        return int(self.candidate.x - self.reference.x)

    @property
    def nominal_dy(self) -> int:
        return int(self.candidate.y - self.reference.y)

    @property
    def stage_dx(self) -> int:
        return int(self.candidate.stage_x - self.reference.stage_x)

    @property
    def stage_dy(self) -> int:
        return int(self.candidate.stage_y - self.reference.stage_y)


@dataclass(frozen=True, slots=True)
class DigitalSlideCalibrationEstimate:
    axis: str
    primary_stride_px: float
    cross_axis_drift_px: float
    pixels_per_step: float | None
    suggested_stage_step: int | None
    confidence: float
    sample_count: int
    accepted_count: int
    source_frame_size: tuple[int, int]
    target_frame_size: tuple[int, int]
    directional_difference_px: float = 0.0
    warnings: tuple[str, ...] = ()

    @property
    def can_apply_pixel_stride(self) -> bool:
        return (
            self.primary_stride_px > 0
            and self.accepted_count > 0
            and min(*self.source_frame_size, *self.target_frame_size) > 0
        )

    @property
    def can_apply_stage_step(self) -> bool:
        return (
            self.suggested_stage_step is not None
            and self.pixels_per_step is not None
            and self.pixels_per_step > 0
            and self.accepted_count >= 2
            and self.confidence >= 0.12
            and self.directional_difference_px <= max(2.0, self.primary_stride_px * 0.05)
        )


class DigitalSlideCalibrationSession:
    """Bounded, read-only access to adjacent fields in one digital slide."""

    def __init__(self, *, cache: DigitalSlideSessionCache | None = None) -> None:
        self._cache = cache or DigitalSlideSessionCache()
        self._store: DigitalSlideStore | None = None
        self._manifest: DigitalSlideManifest | None = None
        self._source_path: Path | None = None
        self._working_path: Path | None = None
        self._descriptor_cache: dict[int, list[DigitalSlideTileDescriptor]] = {}

    @property
    def manifest(self) -> DigitalSlideManifest:
        if self._manifest is None:
            raise RuntimeError("尚未打开数字化切片。")
        return self._manifest

    @property
    def source_path(self) -> Path | None:
        return self._source_path

    @property
    def working_path(self) -> Path | None:
        return self._working_path

    def open(
        self,
        path: str | Path,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
        cancellation_requested: Callable[[], bool] | None = None,
    ) -> DigitalSlideManifest:
        self.close_store()
        source = Path(path).expanduser()
        working = self._cache.localize(
            source,
            progress_callback=progress_callback,
            cancellation_requested=cancellation_requested,
        )
        store = DigitalSlideStore(working)
        try:
            store.open_read_only()
            manifest = store.read_manifest()
        except Exception:
            store.close()
            raise
        self._source_path = source
        self._working_path = working
        self._store = store
        self._manifest = manifest
        self._descriptor_cache.clear()
        return manifest

    def close_store(self) -> None:
        store = self._store
        self._store = None
        self._manifest = None
        self._descriptor_cache.clear()
        if store is not None:
            store.close()

    def close(self) -> None:
        self.close_store()
        self._cache.cleanup()

    def descriptors(self, focus_index: int) -> list[DigitalSlideTileDescriptor]:
        index = int(focus_index)
        cached = self._descriptor_cache.get(index)
        if cached is not None:
            return list(cached)
        if self._store is None:
            return []
        descriptors = self._store.list_tile_descriptors(z_index=index)
        self._descriptor_cache[index] = descriptors
        return list(descriptors)

    def adjacent_pairs(self, focus_index: int, axis: str) -> list[DigitalSlideCalibrationPair]:
        axis = CALIBRATION_AXIS_Y if axis == CALIBRATION_AXIS_Y else CALIBRATION_AXIS_X
        descriptors = self.descriptors(focus_index)
        grouped: dict[int, list[DigitalSlideTileDescriptor]] = defaultdict(list)
        if axis == CALIBRATION_AXIS_X:
            for descriptor in descriptors:
                grouped[int(descriptor.y)].append(descriptor)
            order_key = lambda item: (item.x, item.tile_id)
        else:
            for descriptor in descriptors:
                grouped[int(descriptor.x)].append(descriptor)
            order_key = lambda item: (item.y, item.tile_id)
        pairs: list[DigitalSlideCalibrationPair] = []
        for group_key in sorted(grouped):
            ordered = sorted(grouped[group_key], key=order_key)
            for reference, candidate in zip(ordered, ordered[1:]):
                primary_delta = (
                    candidate.x - reference.x
                    if axis == CALIBRATION_AXIS_X
                    else candidate.y - reference.y
                )
                if primary_delta <= 0:
                    continue
                pairs.append(DigitalSlideCalibrationPair(reference, candidate, axis))
        return pairs

    def read_pair(self, pair: DigitalSlideCalibrationPair) -> tuple[QImage, QImage]:
        if self._store is None:
            return QImage(), QImage()
        return (
            self._store.read_tile_image(pair.reference.tile_id),
            self._store.read_tile_image(pair.candidate.tile_id),
        )

    @staticmethod
    def _sample_pairs(
        pairs: list[DigitalSlideCalibrationPair],
        maximum: int,
    ) -> list[DigitalSlideCalibrationPair]:
        maximum = max(1, min(int(maximum), 12))
        if len(pairs) <= maximum:
            return list(pairs)
        if maximum == 1:
            return [pairs[len(pairs) // 2]]
        return [
            pairs[round(index * (len(pairs) - 1) / (maximum - 1))]
            for index in range(maximum)
        ]

    def estimate(
        self,
        *,
        focus_index: int,
        axis: str,
        target_frame_size: tuple[int, int],
        target_overlap_percent: int,
        current_stage_step: int,
        maximum_pairs: int = 10,
        cancellation_requested: Callable[[], bool] | None = None,
    ) -> DigitalSlideCalibrationEstimate:
        axis = CALIBRATION_AXIS_Y if axis == CALIBRATION_AXIS_Y else CALIBRATION_AXIS_X
        pairs = self._sample_pairs(self.adjacent_pairs(focus_index, axis), maximum_pairs)
        if not pairs:
            raise ValueError("当前焦层在所选方向没有相邻视场。")
        accepted: list[tuple[DigitalSlideCalibrationPair, float, float, float]] = []
        rejection_reasons: list[str] = []
        for pair in pairs:
            if cancellation_requested is not None and cancellation_requested():
                raise RuntimeError("校准估算已取消。")
            reference, candidate = self.read_pair(pair)
            result = estimate_tile_translation(
                reference,
                candidate,
                coarse_dx=float(pair.nominal_dx),
                coarse_dy=float(pair.nominal_dy),
            )
            if result.accepted:
                accepted.append((pair, float(result.dx), float(result.dy), float(result.confidence)))
            else:
                rejection_reasons.append(result.reason)
        if not accepted:
            reason_labels = {
                "overlap": "低重叠",
                "ambiguous": "重复纹理导致匹配不唯一",
                "registration": "低纹理或低置信度",
                "empty": "空图像",
            }
            reason = "、".join(
                reason_labels.get(item, item)
                for item in sorted(set(rejection_reasons))
            ) or "纹理或重叠不足"
            raise ValueError(f"自动估算未获得可靠结果：{reason}。可改用手动微调。")

        primary_values = [abs(dx if axis == CALIBRATION_AXIS_X else dy) for _pair, dx, dy, _confidence in accepted]
        cross_values = [dy if axis == CALIBRATION_AXIS_X else dx for _pair, dx, dy, _confidence in accepted]
        source_width = int(accepted[0][0].reference.width)
        source_height = int(accepted[0][0].reference.height)
        target_width = int(target_frame_size[0])
        target_height = int(target_frame_size[1])
        if min(source_width, source_height, target_width, target_height) <= 0:
            raise ValueError("无法确认源视场或目标采集尺寸，不能直接应用校准结果。")
        primary_scale = target_width / source_width if axis == CALIBRATION_AXIS_X else target_height / source_height
        cross_scale = target_height / source_height if axis == CALIBRATION_AXIS_X else target_width / source_width
        primary_stride = float(median(primary_values)) * primary_scale
        cross_drift = float(median(cross_values)) * cross_scale

        ratios: list[float] = []
        direction_groups: dict[int, list[float]] = defaultdict(list)
        for pair, dx, dy, _confidence in accepted:
            primary = abs(dx if axis == CALIBRATION_AXIS_X else dy)
            stage_delta = pair.stage_dx if axis == CALIBRATION_AXIS_X else pair.stage_dy
            if stage_delta:
                ratios.append(primary / abs(stage_delta))
                # Spatial pairs are always normalized left-to-right/top-to-bottom.
                # Tile IDs retain write order, so their relative order tells us
                # which serpentine pass acquired the pair in reverse.
                traversal_direction = 1 if pair.candidate.tile_id > pair.reference.tile_id else -1
                direction_groups[traversal_direction].append(primary)
        pixels_per_step = float(median(ratios)) * primary_scale if ratios else None
        suggested_stage_step: int | None = None
        if pixels_per_step and pixels_per_step > 0:
            frame_axis = target_width if axis == CALIBRATION_AXIS_X else target_height
            overlap = max(0, min(90, int(target_overlap_percent))) / 100.0
            target_stride = max(1.0, frame_axis * (1.0 - overlap))
            magnitude = max(1, int(round(target_stride / pixels_per_step)))
            suggested_stage_step = -magnitude if int(current_stage_step) < 0 else magnitude

        directional_difference = 0.0
        if len(direction_groups) > 1:
            medians = [float(median(group)) * primary_scale for group in direction_groups.values()]
            directional_difference = max(medians) - min(medians)
        warnings: list[str] = []
        if len(accepted) < len(pairs):
            warnings.append(f"{len(pairs) - len(accepted)} 个样本因纹理或重叠不足被忽略")
        if len(accepted) == 1:
            warnings.append("仅有一个可靠样本，电机步距建议仅供参考")
        if pixels_per_step is None:
            warnings.append("切片缺少有效电机位移，只能校准像素步距")
        if directional_difference > max(2.0, primary_stride * 0.05):
            warnings.append("正反方向结果差异较大，可能存在回程间隙")
        confidence = float(median([confidence for _pair, _dx, _dy, confidence in accepted]))
        if confidence < 0.12:
            warnings.append("自动估算置信度不足，只能使用手动像素校准")
        return DigitalSlideCalibrationEstimate(
            axis=axis,
            primary_stride_px=primary_stride,
            cross_axis_drift_px=cross_drift,
            pixels_per_step=pixels_per_step,
            suggested_stage_step=suggested_stage_step,
            confidence=confidence,
            sample_count=len(pairs),
            accepted_count=len(accepted),
            source_frame_size=(source_width, source_height),
            target_frame_size=(target_width, target_height),
            directional_difference_px=directional_difference,
            warnings=tuple(warnings),
        )


__all__ = [
    "CALIBRATION_AXIS_X",
    "CALIBRATION_AXIS_Y",
    "DigitalSlideCalibrationEstimate",
    "DigitalSlideCalibrationPair",
    "DigitalSlideCalibrationSession",
]
