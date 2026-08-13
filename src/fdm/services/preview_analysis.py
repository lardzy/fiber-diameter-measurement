from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from time import perf_counter
from typing import Any

from PySide6.QtGui import QImage

from fdm.runtime_logging import aggregate_runtime_metric
from fdm.settings import FocusStackProfile

MAP_BUILD_ANALYSIS_INTERVAL_MS = 90
MAP_BUILD_STABLE_REQUIRED_FRAMES = 2
MAP_BUILD_MAX_TILE_FRAMES = 3
MAP_BUILD_PREVIEW_REFRESH_INTERVAL_MS = 250
MOSAIC_RENDER_STRIP_HEIGHT = 256


MIB = 1024 * 1024
GIB = 1024 * MIB


@dataclass(frozen=True, slots=True)
class AnalysisResourceLimits:
    focus_max_frames: int = 12
    focus_max_retained_bytes: int = 256 * MIB
    focus_max_render_working_bytes: int = GIB
    map_max_tiles: int = 32
    map_max_retained_bytes: int = 256 * MIB
    map_max_pixels: int = 32_000_000
    map_max_dimension: int = 10_000
    map_max_render_working_bytes: int = GIB

    def normalized(self) -> "AnalysisResourceLimits":
        return AnalysisResourceLimits(
            focus_max_frames=max(1, int(self.focus_max_frames)),
            focus_max_retained_bytes=max(MIB, int(self.focus_max_retained_bytes)),
            focus_max_render_working_bytes=max(MIB, int(self.focus_max_render_working_bytes)),
            map_max_tiles=max(2, int(self.map_max_tiles)),
            map_max_retained_bytes=max(MIB, int(self.map_max_retained_bytes)),
            map_max_pixels=max(1, int(self.map_max_pixels)),
            map_max_dimension=max(1, int(self.map_max_dimension)),
            map_max_render_working_bytes=max(MIB, int(self.map_max_render_working_bytes)),
        )


DEFAULT_ANALYSIS_RESOURCE_LIMITS = AnalysisResourceLimits()


def _ensure_cv_numpy():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependency guarded at runtime
        raise RuntimeError("opencv-python is required for preview analysis.") from exc
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency guarded at runtime
        raise RuntimeError("numpy is required for preview analysis.") from exc
    return cv2, np


def qimage_to_bgr_array(image: QImage):
    cv2, np = _ensure_cv_numpy()
    if image.isNull():
        raise RuntimeError("当前分析帧为空。")
    rgb = image.convertToFormat(QImage.Format.Format_RGB888)
    buffer = rgb.constBits()
    array = np.frombuffer(buffer, dtype=np.uint8, count=rgb.sizeInBytes())
    array = array.reshape((rgb.height(), rgb.bytesPerLine()))
    rgb_array = array[:, : rgb.width() * 3].reshape((rgb.height(), rgb.width(), 3)).copy()
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)


def bgr_array_to_qimage(array) -> QImage:
    cv2, np = _ensure_cv_numpy()
    image_array = np.clip(array, 0, 255).astype(np.uint8, copy=False)
    rgb_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    height, width = rgb_array.shape[:2]
    bytes_per_line = width * 3
    image = QImage(rgb_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
    return image.copy()


@dataclass(slots=True)
class FocusStackReport:
    preview_image: QImage
    sampled_frames: int
    accepted_frames: int
    message: str
    low_confidence: bool = False
    limit_reached: bool = False
    limit_reason: str = ""
    retained_bytes: int = 0


@dataclass(slots=True)
class FocusStackFinalResult:
    image: QImage
    sampled_frames: int
    accepted_frames: int
    metadata: dict[str, Any]


@dataclass(slots=True)
class MapBuildReport:
    preview_image: QImage
    sampled_frames: int
    accepted_frames: int
    tile_count: int
    message: str
    low_confidence: bool = False
    motion_state: str = "moving"
    stable_streak: int = 0
    translation_px: float = 0.0
    correlation_response: float = 0.0
    quality_score: float = 0.0
    limit_reached: bool = False
    limit_reason: str = ""
    retained_bytes: int = 0
    estimated_output_pixels: int = 0


@dataclass(slots=True)
class MapBuildFinalResult:
    image: QImage
    sampled_frames: int
    accepted_frames: int
    tile_count: int
    metadata: dict[str, Any]


@dataclass(slots=True)
class FocusStackRenderConfig:
    profile: str = FocusStackProfile.BALANCED
    sharpen_strength: int = 35

    def normalized_copy(self) -> "FocusStackRenderConfig":
        profile = self.profile if self.profile in {
            FocusStackProfile.SHARP,
            FocusStackProfile.BALANCED,
            FocusStackProfile.SOFT,
        } else FocusStackProfile.BALANCED
        sharpen_strength = max(0, min(100, int(round(self.sharpen_strength))))
        return FocusStackRenderConfig(
            profile=profile,
            sharpen_strength=sharpen_strength,
        )


@dataclass(slots=True)
class _PreparedFrame:
    bgr: Any
    gray: Any
    focus_map: Any
    small_gray: Any
    sharpness: float


@dataclass(frozen=True, slots=True)
class _FrameFusionResult:
    bgr: Any | None
    limit_reached: bool = False
    limit_reason: str = ""


@dataclass(slots=True)
class _MapMotionFrame:
    image: QImage
    small_gray: Any
    full_shape: tuple[int, int]
    sharpness: float
    prepared: _PreparedFrame | None = None


@dataclass(slots=True)
class _TileRecord:
    tile_id: int
    bgr: Any
    gray: Any
    x: float
    y: float

    @property
    def width(self) -> int:
        return int(self.bgr.shape[1])

    @property
    def height(self) -> int:
        return int(self.bgr.shape[0])


@dataclass(slots=True)
class _TileEdge:
    source_id: int
    target_id: int
    dx: float
    dy: float
    weight: float


@dataclass(slots=True)
class _MapRegistrationConfig:
    min_overlap: float = 0.15
    max_overlap: float = 0.95
    min_phase_response: float = 0.08
    min_ncc: float = 0.55
    min_texture_std: float = 4.0
    max_seed_correction_px: float = 56.0
    min_edge_weight: float = 0.12
    ambiguity_margin: float = 0.08
    ambiguity_distance_px: float = 18.0

    def as_metadata(self) -> dict[str, float]:
        return {
            "min_overlap": self.min_overlap,
            "max_overlap": self.max_overlap,
            "min_phase_response": self.min_phase_response,
            "min_ncc": self.min_ncc,
            "min_texture_std": self.min_texture_std,
            "max_seed_correction_px": self.max_seed_correction_px,
            "min_edge_weight": self.min_edge_weight,
            "ambiguity_margin": self.ambiguity_margin,
            "ambiguity_distance_px": self.ambiguity_distance_px,
        }


@dataclass(slots=True)
class _RegistrationCandidate:
    dx: float
    dy: float
    response: float
    ncc: float
    overlap: float
    seed_delta: float
    score: float


@dataclass(slots=True)
class _RegistrationResult:
    accepted: bool
    dx: float = 0.0
    dy: float = 0.0
    response: float = 0.0
    ncc: float = 0.0
    overlap: float = 0.0
    weight: float = 0.0
    reason: str = "registration"


class FocusAccumulator:
    def __init__(self, *, limits: AnalysisResourceLimits | None = None) -> None:
        self._best_bgr = None
        self._best_focus_map = None
        self._profile_weighted_numerators: dict[str, Any] = {}
        self._profile_weighted_denominators: dict[str, Any] = {}
        self._last_small_gray = None
        self._last_sharpness = 0.0
        self._limits = (limits or DEFAULT_ANALYSIS_RESOURCE_LIMITS).normalized()
        self.sampled_frames = 0
        self.accepted_frames = 0
        self.limit_reached = False
        self.limit_reason = ""
        self.retained_bytes = 0

    def has_frames(self) -> bool:
        return self._best_bgr is not None

    def add_qimage(self, image: QImage) -> bool:
        frame = _prepare_frame(image)
        return self.add_prepared_frame(frame)

    def add_prepared_frame(self, frame: _PreparedFrame) -> bool:
        self.sampled_frames += 1
        if self.limit_reached:
            return False
        if self._last_small_gray is not None and _is_duplicate_frame(
            frame,
            self._last_small_gray,
            self._last_sharpness,
        ):
            return False
        if self.accepted_frames >= self._limits.focus_max_frames:
            self._set_limit(f"景深有效帧已达到 {self._limits.focus_max_frames} 张上限")
            return False
        if self._best_bgr is not None and (
            frame.bgr.shape != self._best_bgr.shape
            or frame.focus_map.shape != self._best_focus_map.shape
        ):
            raise ValueError("景深合成帧尺寸不一致。")
        candidate_retained_bytes = self.estimated_retained_bytes_for(frame)
        if candidate_retained_bytes > self._limits.focus_max_retained_bytes:
            self._set_limit(
                f"景深保留数据已达到 {self._limits.focus_max_retained_bytes / MIB:.0f} MiB 上限"
            )
            return False
        estimated_render_bytes = candidate_retained_bytes + (_array_nbytes(frame.bgr) * 8)
        if estimated_render_bytes > self._limits.focus_max_render_working_bytes:
            self._set_limit("景深预计渲染工作集已达到 1 GiB 上限")
            return False
        cv2, np = _ensure_cv_numpy()
        if self._best_bgr is None:
            self._best_bgr = frame.bgr.copy()
            self._best_focus_map = frame.focus_map.copy()
            for profile in (
                FocusStackProfile.SHARP,
                FocusStackProfile.BALANCED,
                FocusStackProfile.SOFT,
            ):
                self._profile_weighted_numerators[profile] = np.zeros_like(
                    frame.bgr,
                    dtype=np.float32,
                )
                self._profile_weighted_denominators[profile] = np.zeros_like(
                    frame.focus_map,
                    dtype=np.float32,
                )
        else:
            better_focus = frame.focus_map > self._best_focus_map
            np.copyto(self._best_bgr, frame.bgr, where=better_focus[..., None])
            np.copyto(self._best_focus_map, frame.focus_map, where=better_focus)
        image_float = frame.bgr.astype(np.float32, copy=False)
        for profile in (
            FocusStackProfile.SHARP,
            FocusStackProfile.BALANCED,
            FocusStackProfile.SOFT,
        ):
            raw_weight = _incremental_profile_raw_weight(frame.focus_map, profile=profile)
            self._profile_weighted_numerators[profile] += image_float * raw_weight[..., None]
            self._profile_weighted_denominators[profile] += raw_weight
        self._last_small_gray = frame.small_gray.copy()
        self._last_sharpness = float(frame.sharpness)
        self.retained_bytes = (
            _array_nbytes(self._best_bgr)
            + _array_nbytes(self._best_focus_map)
            + sum(_array_nbytes(value) for value in self._profile_weighted_numerators.values())
            + sum(_array_nbytes(value) for value in self._profile_weighted_denominators.values())
            + _array_nbytes(self._last_small_gray)
        )
        self.accepted_frames += 1
        return True

    def estimated_retained_bytes_for(self, frame: _PreparedFrame) -> int:
        bgr_bytes = _array_nbytes(frame.bgr)
        focus_bytes = _array_nbytes(frame.focus_map)
        return (
            bgr_bytes
            + focus_bytes
            + (3 * ((bgr_bytes * 4) + focus_bytes))
            + _array_nbytes(frame.small_gray)
        )

    def _set_limit(self, reason: str) -> None:
        self.limit_reached = True
        self.limit_reason = reason

    def render_image(self, render_config: FocusStackRenderConfig | None = None) -> QImage:
        if self._best_bgr is None:
            return QImage()
        config = (render_config or FocusStackRenderConfig()).normalized_copy()
        if self.accepted_frames == 1:
            blended = self._best_bgr.copy()
        else:
            blended = _focus_stack_incremental_render(
                self._best_bgr,
                self._profile_weighted_numerators[config.profile],
                self._profile_weighted_denominators[config.profile],
                profile=config.profile,
            )
        blended = _apply_sharpen_strength(blended, config.sharpen_strength)
        return bgr_array_to_qimage(blended)

    def preview_image(self, render_config: FocusStackRenderConfig | None = None) -> QImage:
        return self.render_image(render_config)

    def final_image(self, render_config: FocusStackRenderConfig | None = None) -> QImage:
        return self.render_image(render_config)

    def latest_sharpness(self) -> float:
        if self._last_small_gray is None:
            return 0.0
        return self._last_sharpness


class FocusStackAnalyzer:
    def __init__(
        self,
        *,
        device_id: str,
        device_name: str,
        render_config: FocusStackRenderConfig | None = None,
        resource_limits: AnalysisResourceLimits | None = None,
    ) -> None:
        self._device_id = device_id
        self._device_name = device_name
        self._accumulator = FocusAccumulator(limits=resource_limits)
        self._render_config = (render_config or FocusStackRenderConfig()).normalized_copy()

    def add_frame(self, image: QImage) -> FocusStackReport:
        accepted = self._accumulator.add_qimage(image)
        preview = self._accumulator.preview_image(self._render_config)
        sampled = self._accumulator.sampled_frames
        accepted_count = self._accumulator.accepted_frames
        message = f"采样 {sampled} 帧 | 接受 {accepted_count} 帧"
        if not accepted:
            message += f" | {self._accumulator.limit_reason or '重复帧已跳过'}"
        return FocusStackReport(
            preview_image=preview,
            sampled_frames=sampled,
            accepted_frames=accepted_count,
            message=message,
            low_confidence=accepted_count < 2,
            limit_reached=self._accumulator.limit_reached,
            limit_reason=self._accumulator.limit_reason,
            retained_bytes=self._accumulator.retained_bytes,
        )

    def refresh_preview(self) -> FocusStackReport:
        sampled = self._accumulator.sampled_frames
        accepted_count = self._accumulator.accepted_frames
        preview = self._accumulator.preview_image(self._render_config)
        message = f"采样 {sampled} 帧 | 接受 {accepted_count} 帧"
        if accepted_count:
            message += " | 预览参数已更新"
        else:
            message += " | 等待采样"
        return FocusStackReport(
            preview_image=preview,
            sampled_frames=sampled,
            accepted_frames=accepted_count,
            message=message,
            low_confidence=accepted_count < 2,
            limit_reached=self._accumulator.limit_reached,
            limit_reason=self._accumulator.limit_reason,
            retained_bytes=self._accumulator.retained_bytes,
        )

    def set_render_config(self, render_config: FocusStackRenderConfig) -> None:
        self._render_config = render_config.normalized_copy()

    def current_render_config(self) -> FocusStackRenderConfig:
        return self._render_config.normalized_copy()

    def finalize(self, *, render_config: FocusStackRenderConfig | None = None) -> FocusStackFinalResult:
        if not self._accumulator.has_frames():
            raise RuntimeError("景深合成未收到有效采样帧。")
        config = (render_config or self._render_config).normalized_copy()
        image = self._accumulator.final_image(config)
        sampled = self._accumulator.sampled_frames
        accepted = self._accumulator.accepted_frames
        metadata = {
            "analysis_mode": "focus_stack",
            "device_id": self._device_id,
            "device_name": self._device_name,
            "sampled_frames": sampled,
            "accepted_frames": accepted,
            "post_sharpen": bool(config.sharpen_strength > 0),
            "focus_stack_profile": config.profile,
            "sharpen_strength": config.sharpen_strength,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        return FocusStackFinalResult(
            image=image,
            sampled_frames=sampled,
            accepted_frames=accepted,
            metadata=metadata,
        )


class MapBuildAnalyzer:
    def __init__(
        self,
        *,
        device_id: str,
        device_name: str,
        resource_limits: AnalysisResourceLimits | None = None,
    ) -> None:
        self._device_id = device_id
        self._device_name = device_name
        self._render_config = FocusStackRenderConfig(
            profile=FocusStackProfile.BALANCED,
            sharpen_strength=0,
        )
        self._registration_config = _MapRegistrationConfig()
        self._resource_limits = (resource_limits or DEFAULT_ANALYSIS_RESOURCE_LIMITS).normalized()
        self._resource_limit_reached = False
        self._resource_limit_reason = ""
        self._sampled_frames = 0
        self._accepted_frames = 0
        self._rejected_moving_frames = 0
        self._rejected_low_confidence_frames = 0
        self._rejected_registration_frames = 0
        self._rejected_overlap_frames = 0
        self._rejected_ambiguous_frames = 0
        self._stable_accept_count = 0
        self._skipped_tile_frames = 0
        self._tiles: list[_TileRecord] = []
        self._edges: list[_TileEdge] = []
        self._current_accumulator = FocusAccumulator(limits=self._resource_limits)
        self._current_origin_small = None
        self._current_predicted_position = (0.0, 0.0)
        self._pending_current_edge: _TileEdge | None = None
        self._last_tile_delta: tuple[float, float] | None = None
        self._tile_counter = 0
        self._previous_frame: _MapMotionFrame | None = None
        self._transition_pending = False
        self._is_stable = False
        self._stable_streak = 0
        self._unstable_streak = 0
        self._stable_window: list[_MapMotionFrame] = []
        self._stable_required = MAP_BUILD_STABLE_REQUIRED_FRAMES
        self._last_message = "等待移动样品台并采样"
        self._stable_step_threshold_px: float | None = None
        self._tile_freeze_threshold_px: float | None = None
        self._stable_response_threshold = 0.015
        self._resume_origin_threshold_px: float | None = None
        self._last_translation_px = 0.0
        self._last_response = 0.0
        self._last_quality_score = 0.0
        self._last_motion_state = "moving"
        self._preview_image_cache = QImage()
        self._preview_dirty = True
        self._preview_render_count = 0
        self._last_preview_render_at = 0.0
        self._last_perf_metrics: dict[str, float | int | bool] = {}

    def add_frame(self, image: QImage) -> MapBuildReport:
        self._reset_perf_metrics()
        if self._resource_limit_reached:
            return self._build_report()
        light_started = perf_counter()
        frame = _prepare_map_motion_frame(image)
        self._last_perf_metrics["light_motion_prep_ms"] = (perf_counter() - light_started) * 1000.0
        self._sampled_frames += 1
        if (
            self._retained_bytes() + _map_motion_frame_bytes(frame)
            > self._resource_limits.map_max_retained_bytes
        ):
            self._set_resource_limit(
                f"地图保留数据已达到 {self._resource_limits.map_max_retained_bytes / MIB:.0f} MiB 上限"
            )
            return self._build_report()
        self._initialize_thresholds(frame)
        self._last_quality_score = frame.sharpness
        if self._previous_frame is None:
            self._previous_frame = frame
            self._stable_window = [frame]
            self._stable_streak = 1
            self._unstable_streak = 0
            self._last_translation_px = 0.0
            self._last_response = 1.0
            self._last_message = self._settling_message()
            self._last_motion_state = "settling"
            return self._build_report()

        motion_started = perf_counter()
        step_phase_dx, step_phase_dy, step_response = _estimate_translation(self._previous_frame.small_gray, frame.small_gray)
        step_scale = _small_frame_scale(frame.full_shape, frame.small_gray.shape)
        step_dx = -step_phase_dx * step_scale
        step_dy = -step_phase_dy * step_scale
        step_translation = math.hypot(step_dx, step_dy)
        self._last_translation_px = step_translation
        self._last_response = step_response
        self._previous_frame = frame

        newly_stable, stable_anchor = self._update_stability_gate(
            frame,
            translation_px=step_translation,
            response=step_response,
            allow_soft_stable=self._is_stable and self._current_accumulator.has_frames(),
        )

        origin_dx = 0.0
        origin_dy = 0.0
        origin_translation = 0.0
        if self._current_origin_small is not None:
            origin_phase_dx, origin_phase_dy, _ = _estimate_translation(self._current_origin_small, frame.small_gray)
            origin_dx = -origin_phase_dx * step_scale
            origin_dy = -origin_phase_dy * step_scale
            origin_translation = math.hypot(origin_dx, origin_dy)
        self._last_perf_metrics["motion_eval_ms"] = (perf_counter() - motion_started) * 1000.0

        if self._current_accumulator.has_frames() and origin_translation <= float(self._resume_origin_threshold_px or 4.0):
            self._transition_pending = False

        if self._current_accumulator.has_frames() and origin_translation >= float(self._tile_freeze_threshold_px or 6.0):
            if not self._transition_pending:
                self._transition_pending = True
                self._is_stable = False
                self._unstable_streak = 0
                if step_translation <= float(self._stable_step_threshold_px or 2.0) and step_response >= self._stable_response_threshold:
                    self._stable_streak = 1
                    self._stable_window = [frame]
                else:
                    self._stable_streak = 0
                    self._stable_window.clear()

        if not self._current_accumulator.has_frames():
            if self._is_stable and stable_anchor is not None:
                self._current_origin_small = stable_anchor.small_gray
                self._current_predicted_position = (0.0, 0.0)
                accept_status = self._accept_motion_frame(stable_anchor)
                if accept_status != "limit":
                    self._last_message = self._sampling_message("开始采样首个 tile")
                    self._last_motion_state = "sampling"
            else:
                self._rejected_moving_frames += 1
                self._last_message = self._motion_wait_message()
                self._last_motion_state = "settling" if self._stable_streak > 0 else "moving"
            return self._build_report()

        if self._transition_pending:
            if not self._is_stable:
                self._rejected_moving_frames += 1
                self._last_message = "检测到新位置，等待静止后再采样候选 tile"
                self._last_motion_state = "moving" if self._stable_streak == 0 else "settling"
                return self._build_report()

            candidate_frames = list(self._stable_window)
            if not candidate_frames:
                candidate = stable_anchor or self._best_stable_frame()
                candidate_frames = [candidate] if candidate is not None else []
            if not candidate_frames:
                self._last_message = "检测到新位置，等待静止后再采样候选 tile"
                self._last_motion_state = "settling"
                return self._build_report()
            self._try_commit_candidate_tile(candidate_frames, coarse_dx=origin_dx, coarse_dy=origin_dy)
            return self._build_report()

        if not self._is_stable:
            self._rejected_moving_frames += 1
            self._last_message = self._motion_wait_message()
            self._last_motion_state = "settling" if self._stable_streak > 0 else "moving"
            return self._build_report()

        candidate = stable_anchor if newly_stable and stable_anchor is not None else frame
        accept_status = self._accept_motion_frame(candidate)
        if accept_status == "accepted":
            self._last_message = self._sampling_message("当前 tile 继续采样")
        elif accept_status == "full":
            self._last_message = self._sampling_message(
                f"当前 tile 已采够 {MAP_BUILD_MAX_TILE_FRAMES} 帧，继续监测移动"
            )
        elif accept_status == "duplicate":
            self._last_message = f"{self._sampling_message()} | 当前帧与上一帧接近，已跳过"
        if accept_status != "limit":
            self._last_motion_state = "sampling"
        return self._build_report()

    def finalize(self) -> MapBuildFinalResult:
        self._finalize_current_tile()
        if not self._tiles:
            raise RuntimeError("地图构建未生成有效 tile。")
        if len(self._tiles) < 2:
            raise RuntimeError("地图构建至少需要两个可靠 tile。")
        if not self._edges:
            raise RuntimeError("重叠纹理不足，未生成可靠地图。")
        self._validate_mosaic_limits(self._tiles)
        image = _render_mosaic(self._tiles)
        metadata = {
            "analysis_mode": "map_build",
            "device_id": self._device_id,
            "device_name": self._device_name,
            "sampled_frames": self._sampled_frames,
            "accepted_frames": self._accepted_frames,
            "tile_count": len(self._tiles),
            "edge_count": len(self._edges),
            "rejected_moving_frames": self._rejected_moving_frames,
            "rejected_low_confidence_frames": self._rejected_low_confidence_frames,
            "rejected_registration_frames": self._rejected_registration_frames,
            "rejected_overlap_frames": self._rejected_overlap_frames,
            "rejected_ambiguous_frames": self._rejected_ambiguous_frames,
            "stable_accept_count": self._stable_accept_count,
            "map_build_interval_ms": MAP_BUILD_ANALYSIS_INTERVAL_MS,
            "stable_required_frames": MAP_BUILD_STABLE_REQUIRED_FRAMES,
            "max_tile_frames": MAP_BUILD_MAX_TILE_FRAMES,
            "preview_refresh_interval_ms": MAP_BUILD_PREVIEW_REFRESH_INTERVAL_MS,
            "preview_render_count": self._preview_render_count,
            "skipped_tile_frames": self._skipped_tile_frames,
            "registration_thresholds": self._registration_config.as_metadata(),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        return MapBuildFinalResult(
            image=bgr_array_to_qimage(image),
            sampled_frames=self._sampled_frames,
            accepted_frames=self._accepted_frames,
            tile_count=len(self._tiles),
            metadata=metadata,
        )

    def _build_report(self) -> MapBuildReport:
        preview_image = self._preview_image_cache
        has_preview_content = bool(self._tiles) or self._current_accumulator.has_frames()
        now = perf_counter()
        elapsed_since_render_ms = (now - self._last_preview_render_at) * 1000.0 if self._last_preview_render_at else math.inf
        should_render = (
            has_preview_content
            and (
                self._preview_dirty
                or preview_image.isNull()
                or elapsed_since_render_ms >= MAP_BUILD_PREVIEW_REFRESH_INTERVAL_MS
            )
        )
        if should_render:
            preview_started = perf_counter()
            preview_image = self._render_preview_image()
            self._last_perf_metrics["preview_render_ms"] = (perf_counter() - preview_started) * 1000.0
            self._last_perf_metrics["preview_rendered"] = True
            self._preview_image_cache = preview_image.copy()
            self._preview_dirty = False
            self._last_preview_render_at = now
            self._preview_render_count += 1
        else:
            self._last_perf_metrics["preview_render_ms"] = 0.0
            self._last_perf_metrics["preview_rendered"] = False
        warning = self._low_confidence_warning()
        message = (
            f"采样 {self._sampled_frames} 帧 | 接受 {self._accepted_frames} 帧 | tile {max(1, len(self._tiles))}"
        )
        if warning:
            message += f" | {warning}"
        elif self._last_message:
            message += f" | {self._last_message}"
        low_confidence = bool(warning or "未创建新 tile" in self._last_message)
        return MapBuildReport(
            preview_image=preview_image.copy() if not preview_image.isNull() else QImage(),
            sampled_frames=self._sampled_frames,
            accepted_frames=self._accepted_frames,
            tile_count=len(self._tiles) + (1 if self._current_accumulator.has_frames() else 0),
            message=message,
            low_confidence=low_confidence,
            motion_state=self._motion_state(),
            stable_streak=self._stable_streak,
            translation_px=self._last_translation_px,
            correlation_response=self._last_response,
            quality_score=self._last_quality_score,
            limit_reached=self._resource_limit_reached,
            limit_reason=self._resource_limit_reason,
            retained_bytes=self._retained_bytes(),
            estimated_output_pixels=self._estimated_output_pixels(),
        )

    def _render_preview_image(self) -> QImage:
        preview_tiles = list(self._tiles)
        if self._current_accumulator.has_frames():
            current_preview = self._current_accumulator.preview_image(self._render_config)
            if not current_preview.isNull():
                current_bgr = qimage_to_bgr_array(current_preview)
                preview_tiles = preview_tiles + [
                    _TileRecord(
                        tile_id=-1,
                        bgr=current_bgr,
                        gray=_to_gray(current_bgr),
                        x=self._current_predicted_position[0],
                        y=self._current_predicted_position[1],
                    )
                ]
        mosaic = _render_mosaic(preview_tiles, max_dimension=2400) if preview_tiles else None
        return bgr_array_to_qimage(mosaic) if mosaic is not None else QImage()

    def _finalize_current_tile(self) -> _TileRecord | None:
        if not self._current_accumulator.has_frames():
            return None
        tile_image = self._current_accumulator.final_image(self._render_config)
        if tile_image.isNull():
            return None
        tile_bgr = qimage_to_bgr_array(tile_image)
        tile_gray = _to_gray(tile_bgr)
        tile = _TileRecord(
            tile_id=self._tile_counter,
            bgr=tile_bgr,
            gray=tile_gray,
            x=self._current_predicted_position[0],
            y=self._current_predicted_position[1],
        )
        candidate_tiles = [*self._tiles, tile]
        candidate_bytes = self._retained_bytes(include_current=False) + _tile_record_bytes(tile)
        if len(candidate_tiles) > self._resource_limits.map_max_tiles:
            self._set_resource_limit(f"地图 tile 已达到 {self._resource_limits.map_max_tiles} 张上限")
            self._discard_current_accumulator()
            return None
        if candidate_bytes > self._resource_limits.map_max_retained_bytes:
            self._set_resource_limit(
                f"地图保留数据已达到 {self._resource_limits.map_max_retained_bytes / MIB:.0f} MiB 上限"
            )
            self._discard_current_accumulator()
            return None
        try:
            self._validate_mosaic_limits(candidate_tiles)
        except RuntimeError as exc:
            self._set_resource_limit(str(exc))
            self._discard_current_accumulator()
            return None
        self._tile_counter += 1
        self._tiles.append(tile)
        if self._pending_current_edge is not None and self._pending_current_edge.target_id == tile.tile_id:
            self._edges.append(self._pending_current_edge)
            self._pending_current_edge = None
        self._optimize_tile_positions()
        self._current_accumulator = FocusAccumulator(limits=self._resource_limits)
        self._current_origin_small = None
        self._mark_preview_dirty()
        return tile

    def _discard_current_accumulator(self) -> None:
        self._current_accumulator = FocusAccumulator(limits=self._resource_limits)
        self._current_origin_small = None
        self._pending_current_edge = None
        self._mark_preview_dirty()

    def _set_resource_limit(self, reason: str) -> None:
        self._resource_limit_reached = True
        self._resource_limit_reason = reason
        self._last_message = reason
        self._last_motion_state = "limit_reached"

    def _retained_bytes(self, *, include_current: bool = True) -> int:
        retained = sum(_tile_record_bytes(tile) for tile in self._tiles)
        if include_current:
            retained += self._current_accumulator.retained_bytes
        seen_frames: set[int] = set()
        for frame in [self._previous_frame, *self._stable_window]:
            if frame is None or id(frame) in seen_frames:
                continue
            seen_frames.add(id(frame))
            retained += _map_motion_frame_bytes(frame)
        if not self._preview_image_cache.isNull():
            retained += max(0, int(self._preview_image_cache.sizeInBytes()))
        return retained

    def _estimated_output_pixels(self) -> int:
        tiles = list(self._tiles)
        if not tiles:
            return 0
        width, height = _mosaic_dimensions(tiles)
        return width * height

    def _validate_mosaic_limits(self, tiles: list[_TileRecord]) -> None:
        width, height = _mosaic_dimensions(tiles)
        if max(width, height) > self._resource_limits.map_max_dimension:
            raise RuntimeError(f"地图最长边超过 {self._resource_limits.map_max_dimension} px 上限")
        pixels = width * height
        if pixels > self._resource_limits.map_max_pixels:
            raise RuntimeError("地图输出超过 32 MP 上限")
        if _estimate_mosaic_render_working_bytes(width, height) > self._resource_limits.map_max_render_working_bytes:
            raise RuntimeError("地图预计渲染工作集超过 1 GiB 上限")

    def _optimize_tile_positions(self) -> None:
        if len(self._tiles) <= 1 or not self._edges:
            return
        cv2, np = _ensure_cv_numpy()
        del cv2
        tile_by_id = {tile.tile_id: tile for tile in self._tiles}
        anchor_id = self._tiles[0].tile_id
        solve_ids = [tile.tile_id for tile in self._tiles if tile.tile_id != anchor_id]
        index_map = {tile_id: index for index, tile_id in enumerate(solve_ids)}
        if not index_map:
            return
        ax: list[list[float]] = []
        bx: list[float] = []
        ay: list[list[float]] = []
        by: list[float] = []
        for edge in self._edges:
            row_x = [0.0] * len(index_map)
            row_y = [0.0] * len(index_map)
            if edge.target_id != anchor_id:
                row_x[index_map[edge.target_id]] += edge.weight
                row_y[index_map[edge.target_id]] += edge.weight
            if edge.source_id != anchor_id:
                row_x[index_map[edge.source_id]] -= edge.weight
                row_y[index_map[edge.source_id]] -= edge.weight
            bx.append(edge.dx * edge.weight)
            by.append(edge.dy * edge.weight)
            ax.append(row_x)
            ay.append(row_y)
        if not ax:
            return
        x_solution, *_ = np.linalg.lstsq(np.array(ax, dtype=np.float32), np.array(bx, dtype=np.float32), rcond=None)
        y_solution, *_ = np.linalg.lstsq(np.array(ay, dtype=np.float32), np.array(by, dtype=np.float32), rcond=None)
        tile_by_id[anchor_id].x = 0.0
        tile_by_id[anchor_id].y = 0.0
        for tile_id, index in index_map.items():
            tile_by_id[tile_id].x = float(x_solution[index])
            tile_by_id[tile_id].y = float(y_solution[index])

    def _low_confidence_warning(self) -> str:
        if self._tiles and not self._edges and len(self._tiles) > 1:
            return "重叠纹理不足，未生成可靠地图"
        if self._edges and self._edges[-1].weight <= 0.08:
            return "最近 tile 匹配置信度较低"
        return ""

    def _motion_state(self) -> str:
        return self._last_motion_state

    def _settling_message(self) -> str:
        return f"静止确认中 {min(self._stable_streak, self._stable_required)}/{self._stable_required}"

    def _sampling_message(self, detail: str = "") -> str:
        message = f"已静止，正在采样 tile {len(self._tiles) + 1}"
        if detail:
            message += f" | {detail}"
        return message

    def _motion_wait_message(self) -> str:
        if self._transition_pending:
            return "检测到新位置，等待静止后再采样候选 tile"
        if self._stable_streak > 0:
            return self._settling_message()
        return "运动中，暂停入图"

    def _initialize_thresholds(self, frame: _MapMotionFrame) -> None:
        short_side = min(frame.full_shape[0], frame.full_shape[1])
        if self._stable_step_threshold_px is None:
            self._stable_step_threshold_px = max(2.0, short_side * 0.005)
        if self._tile_freeze_threshold_px is None:
            self._tile_freeze_threshold_px = max(6.0, short_side * 0.015)
        if self._resume_origin_threshold_px is None:
            self._resume_origin_threshold_px = max(4.0, float(self._tile_freeze_threshold_px or 6.0) * 0.65)

    def _try_commit_candidate_tile(self, candidate_frames: list[_MapMotionFrame], *, coarse_dx: float, coarse_dy: float) -> None:
        registration_started = perf_counter()
        reference_tile = self._current_tile_preview_record()
        prepared_candidates: list[_PreparedFrame] = []
        for candidate in candidate_frames:
            prepared = self._promote_motion_frame(candidate)
            if prepared is None:
                return
            prepared_candidates.append(prepared)
        fusion = _fuse_prepared_frames(
            prepared_candidates,
            self._render_config,
            limits=self._resource_limits,
        )
        if fusion.limit_reached:
            self._last_perf_metrics["registration_ms"] = (
                perf_counter() - registration_started
            ) * 1000.0
            reason = fusion.limit_reason.replace("景深", "地图候选 tile", 1)
            self._set_resource_limit(reason or "地图候选 tile 融合已达到资源上限")
            return
        candidate_bgr = fusion.bgr
        if reference_tile is None or candidate_bgr is None:
            self._last_perf_metrics["registration_ms"] = (perf_counter() - registration_started) * 1000.0
            self._reject_candidate("registration", "候选 tile 图像为空，未创建新 tile")
            return
        candidate_gray = _to_gray(candidate_bgr)
        candidate_tile = _TileRecord(
            tile_id=-2,
            bgr=candidate_bgr,
            gray=candidate_gray,
            x=0.0,
            y=0.0,
        )
        registration = _register_tile_translation(
            reference_tile,
            candidate_tile,
            config=self._registration_config,
            coarse_dx=coarse_dx,
            coarse_dy=coarse_dy,
            last_delta=self._last_tile_delta,
        )
        self._last_perf_metrics["registration_ms"] = (perf_counter() - registration_started) * 1000.0
        if not registration.accepted:
            if registration.reason == "overlap":
                self._reject_candidate("overlap", "候选位置重叠不在 15%-95% 范围内，未创建新 tile")
            elif registration.reason == "ambiguous":
                self._reject_candidate("ambiguous", "候选位置纹理重复，匹配不唯一，未创建新 tile")
            else:
                self._reject_candidate("registration", "候选位置纹理不足或匹配置信度低，未创建新 tile")
            return

        previous_tile = self._finalize_current_tile()
        if previous_tile is None:
            self._reject_candidate("registration", "当前 tile 尚未生成有效图像，未创建新 tile")
            return
        target_id = self._tile_counter
        self._pending_current_edge = _TileEdge(
            source_id=previous_tile.tile_id,
            target_id=target_id,
            dx=registration.dx,
            dy=registration.dy,
            weight=registration.weight,
        )
        self._current_accumulator = FocusAccumulator(limits=self._resource_limits)
        accepted_any = False
        for candidate in prepared_candidates:
            accepted_any = self._accept_prepared_frame(candidate) == "accepted" or accepted_any
        anchor = self._best_prepared_frame(prepared_candidates)
        self._current_origin_small = anchor.small_gray if anchor is not None else candidate_frames[-1].small_gray
        self._current_predicted_position = (
            previous_tile.x + registration.dx,
            previous_tile.y + registration.dy,
        )
        self._transition_pending = False
        self._last_tile_delta = (registration.dx, registration.dy)
        self._last_response = registration.response
        self._last_motion_state = "tile_committed"
        detail = (
            f"创建新 tile，重叠 {registration.overlap:.0%}，"
            f"NCC {registration.ncc:.2f}，response {registration.response:.2f}"
        )
        if not accepted_any:
            detail += " | 候选帧与上一帧接近"
        self._last_message = self._sampling_message(detail)

    def _current_tile_preview_record(self) -> _TileRecord | None:
        if not self._current_accumulator.has_frames():
            return None
        current_preview = self._current_accumulator.final_image(self._render_config)
        if current_preview.isNull():
            return None
        current_bgr = qimage_to_bgr_array(current_preview)
        return _TileRecord(
            tile_id=-1,
            bgr=current_bgr,
            gray=_to_gray(current_bgr),
            x=self._current_predicted_position[0],
            y=self._current_predicted_position[1],
        )

    def _reject_candidate(self, reason: str, message: str) -> None:
        self._rejected_low_confidence_frames += 1
        if reason == "overlap":
            self._rejected_overlap_frames += 1
        elif reason == "ambiguous":
            self._rejected_ambiguous_frames += 1
            self._rejected_registration_frames += 1
        else:
            self._rejected_registration_frames += 1
        self._last_message = message
        self._last_motion_state = "candidate_rejected"
        self._mark_preview_dirty()

    def _best_stable_frame(self) -> _MapMotionFrame | None:
        if not self._stable_window:
            return None
        return max(self._stable_window, key=lambda frame: frame.sharpness)

    def _best_prepared_frame(self, frames: list[_PreparedFrame]) -> _PreparedFrame | None:
        if not frames:
            return None
        return max(frames, key=lambda frame: frame.sharpness)

    def _update_stability_gate(
        self,
        frame: _MapMotionFrame,
        *,
        translation_px: float,
        response: float,
        allow_soft_stable: bool,
    ) -> tuple[bool, _MapMotionFrame | None]:
        stable_threshold = float(self._stable_step_threshold_px or 2.0)
        stationary = translation_px <= stable_threshold and response >= self._stable_response_threshold
        soft_stationary = allow_soft_stable and translation_px <= stable_threshold * 0.55
        if stationary or soft_stationary:
            self._stable_streak += 1
            self._unstable_streak = 0
            self._stable_window.append(frame)
            max_window = max(MAP_BUILD_MAX_TILE_FRAMES, self._stable_required + 1)
            if len(self._stable_window) > max_window:
                self._stable_window = self._stable_window[-max_window:]
            newly_stable = not self._is_stable and self._stable_streak >= self._stable_required
            if newly_stable:
                self._is_stable = True
                return True, self._best_stable_frame()
            return False, None
        self._unstable_streak += 1
        if self._is_stable and self._unstable_streak < 2 and translation_px <= stable_threshold * 0.55:
            return False, None
        self._is_stable = False
        self._stable_streak = 0
        self._stable_window.clear()
        return False, None

    def _accept_motion_frame(self, frame: _MapMotionFrame) -> str:
        if self._current_accumulator.accepted_frames >= MAP_BUILD_MAX_TILE_FRAMES:
            self._skipped_tile_frames += 1
            return "full"
        prepared = self._promote_motion_frame(frame)
        if prepared is None:
            return "limit"
        return self._accept_prepared_frame(prepared)

    def _accept_prepared_frame(self, frame: _PreparedFrame) -> str:
        if self._current_accumulator.accepted_frames >= MAP_BUILD_MAX_TILE_FRAMES:
            self._skipped_tile_frames += 1
            return "full"
        prospective_accumulator_bytes = (
            self._current_accumulator.retained_bytes
            if self._current_accumulator.has_frames()
            else self._current_accumulator.estimated_retained_bytes_for(frame)
        )
        retained_without_accumulator = max(
            0,
            self._retained_bytes() - self._current_accumulator.retained_bytes,
        )
        if (
            retained_without_accumulator + prospective_accumulator_bytes
            > self._resource_limits.map_max_retained_bytes
        ):
            self._set_resource_limit(
                f"地图保留数据已达到 {self._resource_limits.map_max_retained_bytes / MIB:.0f} MiB 上限"
            )
            return "limit"
        accepted = self._current_accumulator.add_prepared_frame(frame)
        if accepted:
            self._accepted_frames += 1
            self._stable_accept_count += 1
            self._mark_preview_dirty()
            return "accepted"
        if self._current_accumulator.limit_reached:
            self._set_resource_limit(
                self._current_accumulator.limit_reason.replace("景深", "地图 tile", 1)
            )
            return "limit"
        return "duplicate"

    def _promote_motion_frame(self, frame: _MapMotionFrame) -> _PreparedFrame | None:
        if frame.prepared is None:
            height, width = frame.full_shape
            estimated_prepared_bytes = max(0, int(height) * int(width) * 8)
            if (
                self._retained_bytes() + estimated_prepared_bytes
                > self._resource_limits.map_max_retained_bytes
            ):
                self._set_resource_limit(
                    f"地图保留数据已达到 {self._resource_limits.map_max_retained_bytes / MIB:.0f} MiB 上限"
                )
                return None
            started = perf_counter()
            prepared = _prepare_frame(frame.image)
            if (
                self._retained_bytes() + _prepared_frame_bytes(prepared)
                > self._resource_limits.map_max_retained_bytes
            ):
                self._set_resource_limit(
                    f"地图保留数据已达到 {self._resource_limits.map_max_retained_bytes / MIB:.0f} MiB 上限"
                )
                return None
            frame.prepared = prepared
            self._last_perf_metrics["full_frame_promote_ms"] = (
                float(self._last_perf_metrics.get("full_frame_promote_ms", 0.0))
                + (perf_counter() - started) * 1000.0
            )
        return frame.prepared

    def _mark_preview_dirty(self) -> None:
        self._preview_dirty = True

    def _reset_perf_metrics(self) -> None:
        self._last_perf_metrics = {
            "light_motion_prep_ms": 0.0,
            "motion_eval_ms": 0.0,
            "full_frame_promote_ms": 0.0,
            "registration_ms": 0.0,
            "preview_render_ms": 0.0,
            "preview_rendered": False,
        }

    def last_performance_metrics(self) -> dict[str, float | int | bool]:
        return dict(self._last_perf_metrics)


def _fuse_prepared_frames(
    frames: list[_PreparedFrame],
    render_config: FocusStackRenderConfig,
    *,
    limits: AnalysisResourceLimits | None = None,
) -> _FrameFusionResult:
    if not frames:
        return _FrameFusionResult(None)
    accumulator = FocusAccumulator(limits=limits)
    for frame in frames:
        accumulator.add_prepared_frame(frame)
        if accumulator.limit_reached:
            return _FrameFusionResult(
                None,
                limit_reached=True,
                limit_reason=accumulator.limit_reason,
            )
    if not accumulator.has_frames():
        return _FrameFusionResult(None)
    image = accumulator.final_image(render_config)
    if image.isNull():
        return _FrameFusionResult(None)
    return _FrameFusionResult(qimage_to_bgr_array(image))


def _register_tile_translation(
    reference: _TileRecord,
    candidate: _TileRecord,
    *,
    config: _MapRegistrationConfig,
    coarse_dx: float,
    coarse_dy: float,
    last_delta: tuple[float, float] | None,
) -> _RegistrationResult:
    seeds = _registration_seed_candidates(
        reference.width,
        reference.height,
        candidate.width,
        candidate.height,
        coarse_dx=coarse_dx,
        coarse_dy=coarse_dy,
        last_delta=last_delta,
        config=config,
    )
    candidates: list[_RegistrationCandidate] = []
    overlap_rejections = 0
    texture_rejections = 0
    for seed_dx, seed_dy in seeds:
        refined = _refine_registration_seed(reference.gray, candidate.gray, seed_dx, seed_dy, config)
        if refined is None:
            overlap_rejections += 1
            continue
        if refined.ncc < config.min_ncc or refined.response < config.min_phase_response:
            texture_rejections += 1
            continue
        candidates.append(refined)
    if not candidates:
        reason = "overlap" if overlap_rejections and not texture_rejections else "registration"
        return _RegistrationResult(accepted=False, reason=reason)

    candidates.sort(key=lambda item: item.score, reverse=True)
    best = candidates[0]
    for other in candidates[1:]:
        if math.hypot(best.dx - other.dx, best.dy - other.dy) < config.ambiguity_distance_px:
            continue
        if other.score >= best.score - config.ambiguity_margin:
            return _RegistrationResult(
                accepted=False,
                dx=best.dx,
                dy=best.dy,
                response=best.response,
                ncc=best.ncc,
                overlap=best.overlap,
                reason="ambiguous",
            )
    weight = max(config.min_edge_weight, min(1.0, best.score))
    return _RegistrationResult(
        accepted=True,
        dx=best.dx,
        dy=best.dy,
        response=best.response,
        ncc=best.ncc,
        overlap=best.overlap,
        weight=weight,
    )


def _registration_seed_candidates(
    width_a: int,
    height_a: int,
    width_b: int,
    height_b: int,
    *,
    coarse_dx: float,
    coarse_dy: float,
    last_delta: tuple[float, float] | None,
    config: _MapRegistrationConfig,
) -> list[tuple[float, float]]:
    del width_b, height_b
    seeds: list[tuple[float, float]] = []

    def add(dx: float, dy: float) -> None:
        if not math.isfinite(dx) or not math.isfinite(dy):
            return
        rounded = (round(dx, 1), round(dy, 1))
        if rounded not in {(round(seed_dx, 1), round(seed_dy, 1)) for seed_dx, seed_dy in seeds}:
            seeds.append((float(dx), float(dy)))

    if last_delta is not None:
        add(last_delta[0], last_delta[1])
    if math.hypot(coarse_dx, coarse_dy) > 1.0:
        add(coarse_dx, coarse_dy)
        if abs(coarse_dx) >= abs(coarse_dy):
            add(coarse_dx, 0.0)
        else:
            add(0.0, coarse_dy)

    overlap_guesses = (0.20, 0.35, 0.50, 0.80, 0.90)
    for overlap in overlap_guesses:
        shift_x = width_a * (1.0 - overlap)
        shift_y = height_a * (1.0 - overlap)
        add(shift_x, 0.0)
        add(-shift_x, 0.0)
        add(0.0, shift_y)
        add(0.0, -shift_y)

    min_shift_x = width_a * (1.0 - config.max_overlap)
    min_shift_y = height_a * (1.0 - config.max_overlap)
    max_shift_x = width_a * (1.0 - config.min_overlap)
    max_shift_y = height_a * (1.0 - config.min_overlap)
    return [
        (dx, dy)
        for dx, dy in seeds
        if (
            min_shift_x <= abs(dx) <= max_shift_x
            or min_shift_y <= abs(dy) <= max_shift_y
        )
    ]


def _refine_registration_seed(gray_a, gray_b, seed_dx: float, seed_dy: float, config: _MapRegistrationConfig) -> _RegistrationCandidate | None:
    overlap = _predicted_overlap_ratio(gray_a.shape[1], gray_a.shape[0], gray_b.shape[1], gray_b.shape[0], seed_dx, seed_dy)
    if overlap < config.min_overlap or overlap > config.max_overlap:
        return None
    crop_a, crop_b = _crop_overlap_min(gray_a, gray_b, seed_dx, seed_dy, min_size=36)
    if crop_a is None or crop_b is None:
        return None
    if _texture_std(crop_a) < config.min_texture_std or _texture_std(crop_b) < config.min_texture_std:
        return None
    residual_phase_dx, residual_phase_dy, response = _estimate_translation(crop_a, crop_b)
    refined_dx = seed_dx - residual_phase_dx
    refined_dy = seed_dy - residual_phase_dy
    seed_delta = math.hypot(refined_dx - seed_dx, refined_dy - seed_dy)
    if seed_delta > config.max_seed_correction_px:
        return None
    refined_overlap = _predicted_overlap_ratio(
        gray_a.shape[1],
        gray_a.shape[0],
        gray_b.shape[1],
        gray_b.shape[0],
        refined_dx,
        refined_dy,
    )
    if refined_overlap < config.min_overlap or refined_overlap > config.max_overlap:
        return None
    refined_a, refined_b = _crop_overlap_min(gray_a, gray_b, refined_dx, refined_dy, min_size=36)
    if refined_a is None or refined_b is None:
        return None
    ncc = _normalized_cross_correlation(refined_a, refined_b)
    score = (0.72 * max(0.0, ncc)) + (0.28 * max(0.0, min(1.0, response)))
    return _RegistrationCandidate(
        dx=float(refined_dx),
        dy=float(refined_dy),
        response=float(response),
        ncc=float(ncc),
        overlap=float(refined_overlap),
        seed_delta=float(seed_delta),
        score=float(score),
    )


def _prepare_frame(image: QImage) -> _PreparedFrame:
    cv2, np = _ensure_cv_numpy()
    bgr = qimage_to_bgr_array(image)
    gray = _to_gray(bgr)
    focus_map = _focus_measure(gray)
    sharpness = float(focus_map.mean())
    scale = min(1.0, 256.0 / max(gray.shape[0], gray.shape[1]))
    if scale < 1.0:
        small_gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small_gray = gray
    return _PreparedFrame(
        bgr=bgr,
        gray=gray,
        focus_map=focus_map,
        small_gray=small_gray,
        sharpness=sharpness,
    )


def _prepare_map_motion_frame(image: QImage) -> _MapMotionFrame:
    cv2, np = _ensure_cv_numpy()
    if image.isNull():
        raise RuntimeError("当前分析帧为空。")
    gray_image = image.convertToFormat(QImage.Format.Format_Grayscale8)
    buffer = gray_image.constBits()
    array = np.frombuffer(buffer, dtype=np.uint8, count=gray_image.sizeInBytes())
    array = array.reshape((gray_image.height(), gray_image.bytesPerLine()))
    gray = array[:, : gray_image.width()].copy()
    scale = min(1.0, 256.0 / max(gray.shape[0], gray.shape[1]))
    if scale < 1.0:
        small_gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small_gray = gray
    small_focus = _focus_measure(small_gray)
    return _MapMotionFrame(
        image=image.copy(),
        small_gray=small_gray,
        full_shape=gray.shape[:2],
        sharpness=float(small_focus.mean()),
    )


def _to_gray(bgr):
    cv2, _ = _ensure_cv_numpy()
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _focus_measure(gray):
    cv2, np = _ensure_cv_numpy()
    gray_f = gray.astype(np.float32, copy=False)
    lap = cv2.Laplacian(gray_f, cv2.CV_32F, ksize=3)
    sobel_x = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(sobel_x, sobel_y)
    score = cv2.GaussianBlur(np.abs(lap) + 0.35 * grad, (0, 0), sigmaX=1.0, sigmaY=1.0)
    return score


def _is_duplicate_frame(
    frame: _PreparedFrame,
    previous_small_gray,
    previous_sharpness: float,
) -> bool:
    _, np = _ensure_cv_numpy()
    current = frame.small_gray.astype(np.float32, copy=False)
    last = previous_small_gray.astype(np.float32, copy=False)
    diff = float(np.mean(np.abs(current - last)))
    sharpness_delta = abs(frame.sharpness - previous_sharpness) / max(previous_sharpness, 1.0)
    return diff < 1.2 and sharpness_delta < 0.03


def _focus_stack_fast(images: list, focus_maps: list):
    _, np = _ensure_cv_numpy()
    score_stack = np.stack(focus_maps, axis=0)
    winner = np.argmax(score_stack, axis=0)
    image_stack = np.stack(images, axis=0)
    rows, cols = np.indices(winner.shape)
    return image_stack[winner, rows, cols]


def _focus_stack_profile_params(profile: str) -> tuple[tuple[float, ...], float, float]:
    if profile == FocusStackProfile.SHARP:
        return (0.8, 1.6, 3.0), 1.65, 0.72
    if profile == FocusStackProfile.SOFT:
        return (1.8, 4.0, 7.0), 0.9, 0.15
    return (1.0, 2.5, 5.0), 1.2, 0.42


def _incremental_profile_raw_weight(focus_map, *, profile: str):
    """Approximate legacy multi-scale normalization with a constant-size sum.

    The per-frame raw response is averaged across the same Gaussian scales as
    the legacy renderer, then accumulated as numerator/denominator pairs. This
    preserves focus confidence and transition softness without retaining a
    historical frame stack.
    """

    cv2, np = _ensure_cv_numpy()
    sigmas, focus_power, _hard_mix = _focus_stack_profile_params(profile)
    combined = np.zeros_like(focus_map, dtype=np.float32)
    for sigma in sigmas:
        smoothed = cv2.GaussianBlur(focus_map, (0, 0), sigmaX=sigma, sigmaY=sigma)
        combined += np.power(np.clip(smoothed, 1e-6, None), focus_power)
    combined /= max(1.0, float(len(sigmas)))
    combined += 1e-6
    return combined


def _focus_stack_multiscale(images: list, focus_maps: list, *, profile: str = FocusStackProfile.BALANCED):
    cv2, np = _ensure_cv_numpy()
    image_stack = np.stack([image.astype(np.float32, copy=False) for image in images], axis=0)
    total_weights = np.zeros((len(focus_maps),) + focus_maps[0].shape, dtype=np.float32)
    sigmas, focus_power, _hard_mix = _focus_stack_profile_params(profile)
    for sigma in sigmas:
        smoothed = np.stack([cv2.GaussianBlur(focus, (0, 0), sigmaX=sigma, sigmaY=sigma) for focus in focus_maps], axis=0)
        smoothed = np.power(np.clip(smoothed, 1e-6, None), focus_power)
        smoothed += 1e-6
        smoothed /= smoothed.sum(axis=0, keepdims=True)
        total_weights += smoothed
    total_weights /= max(1.0, float(len(sigmas)))
    total_weights /= np.clip(total_weights.sum(axis=0, keepdims=True), 1e-6, None)
    fused = np.sum(image_stack * total_weights[..., None], axis=0)
    return np.clip(fused, 0, 255).astype(np.uint8)


def _focus_stack_render(images: list, focus_maps: list, render_config: FocusStackRenderConfig):
    _, np = _ensure_cv_numpy()
    config = render_config.normalized_copy()
    hard_mix = _focus_stack_profile_params(config.profile)[2]
    hard = _focus_stack_fast(images, focus_maps).astype(np.float32, copy=False)
    soft = _focus_stack_multiscale(images, focus_maps, profile=config.profile).astype(np.float32, copy=False)
    blended = (soft * (1.0 - hard_mix)) + (hard * hard_mix)
    return np.clip(blended, 0, 255).astype(np.uint8)


def _focus_stack_incremental_render(
    best_bgr,
    weighted_numerator,
    weighted_denominator,
    *,
    profile: str = FocusStackProfile.BALANCED,
):
    """Render a focus-confidence weighted incremental approximation."""
    _, np = _ensure_cv_numpy()
    denominator = np.clip(weighted_denominator, 1e-6, None)
    soft = weighted_numerator / denominator[..., None]
    hard_mix = _focus_stack_profile_params(profile)[2]
    blended = (soft * (1.0 - hard_mix)) + (
        best_bgr.astype(np.float32, copy=False) * hard_mix
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def _apply_sharpen_strength(bgr, sharpen_strength: int):
    cv2, np = _ensure_cv_numpy()
    if sharpen_strength <= 0:
        return np.clip(bgr, 0, 255).astype(np.uint8, copy=False)
    amount = max(0.0, min(1.2, float(sharpen_strength) / 100.0))
    blurred = cv2.GaussianBlur(bgr, (0, 0), sigmaX=1.1, sigmaY=1.1)
    sharpened = cv2.addWeighted(
        bgr.astype(np.float32, copy=False),
        1.0 + amount,
        blurred.astype(np.float32, copy=False),
        -amount,
        0.0,
    )
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8, copy=False)
    return sharpened


def _estimate_translation(gray_a, gray_b) -> tuple[float, float, float]:
    cv2, _ = _ensure_cv_numpy()
    a = gray_a.astype("float32", copy=False)
    b = gray_b.astype("float32", copy=False)
    hanning = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(a, b, hanning)
    return float(dx), float(dy), float(response)


def _small_frame_scale(full_shape: tuple[int, int], small_shape: tuple[int, int]) -> float:
    full_h, full_w = full_shape[:2]
    small_h, small_w = small_shape[:2]
    return max(full_h / max(1, small_h), full_w / max(1, small_w))


def _predicted_overlap_ratio(width_a: int, height_a: int, width_b: int, height_b: int, dx: float, dy: float) -> float:
    overlap_w = max(0.0, min(width_a, dx + width_b) - max(0.0, dx))
    overlap_h = max(0.0, min(height_a, dy + height_b) - max(0.0, dy))
    overlap_area = overlap_w * overlap_h
    if overlap_area <= 0:
        return 0.0
    base = min(width_a * height_a, width_b * height_b)
    return float(overlap_area / max(1.0, base))


def _crop_overlap_min(gray_a, gray_b, dx: float, dy: float, *, min_size: int):
    dx_i = int(round(dx))
    dy_i = int(round(dy))
    x1_a = max(0, dx_i)
    y1_a = max(0, dy_i)
    x1_b = max(0, -dx_i)
    y1_b = max(0, -dy_i)
    overlap_w = min(gray_a.shape[1] - x1_a, gray_b.shape[1] - x1_b)
    overlap_h = min(gray_a.shape[0] - y1_a, gray_b.shape[0] - y1_b)
    if overlap_w < min_size or overlap_h < min_size:
        return None, None
    crop_a = gray_a[y1_a : y1_a + overlap_h, x1_a : x1_a + overlap_w]
    crop_b = gray_b[y1_b : y1_b + overlap_h, x1_b : x1_b + overlap_w]
    if crop_a.size == 0 or crop_b.size == 0:
        return None, None
    return crop_a, crop_b


def _texture_std(gray) -> float:
    _, np = _ensure_cv_numpy()
    return float(np.std(gray.astype(np.float32, copy=False)))


def _normalized_cross_correlation(gray_a, gray_b) -> float:
    _, np = _ensure_cv_numpy()
    a = gray_a.astype(np.float32, copy=False)
    b = gray_b.astype(np.float32, copy=False)
    a_centered = a - float(a.mean())
    b_centered = b - float(b.mean())
    denom = float(np.sqrt(np.sum(a_centered * a_centered) * np.sum(b_centered * b_centered)))
    if denom <= 1e-6:
        return -1.0
    return float(np.sum(a_centered * b_centered) / denom)


def _array_nbytes(value: Any) -> int:
    return max(0, int(getattr(value, "nbytes", 0) or 0))


def _prepared_frame_bytes(frame: _PreparedFrame) -> int:
    return sum(
        _array_nbytes(value)
        for value in (frame.bgr, frame.gray, frame.focus_map, frame.small_gray)
    )


def _map_motion_frame_bytes(frame: _MapMotionFrame) -> int:
    image_bytes = 0 if frame.image.isNull() else max(0, int(frame.image.sizeInBytes()))
    retained = image_bytes + _array_nbytes(frame.small_gray)
    if frame.prepared is not None:
        retained += _prepared_frame_bytes(frame.prepared)
    return retained


def _tile_record_bytes(tile: _TileRecord) -> int:
    return _array_nbytes(tile.bgr) + _array_nbytes(tile.gray)


def _mosaic_dimensions(tiles: list[_TileRecord]) -> tuple[int, int]:
    if not tiles:
        return 0, 0
    min_x = math.floor(min(tile.x for tile in tiles))
    min_y = math.floor(min(tile.y for tile in tiles))
    max_x = math.ceil(max(tile.x + tile.width for tile in tiles))
    max_y = math.ceil(max(tile.y + tile.height for tile in tiles))
    return max(1, max_x - min_x), max(1, max_y - min_y)


def _estimate_mosaic_render_working_bytes(
    width: int,
    height: int,
    *,
    strip_height: int = MOSAIC_RENDER_STRIP_HEIGHT,
) -> int:
    width = max(0, int(width))
    height = max(0, int(height))
    if width == 0 or height == 0:
        return 0
    active_rows = min(height, max(1, int(strip_height)))
    output_bytes = width * height * 3
    # RGB accumulator + scalar weights + feather mask + weighted tile crop.
    strip_working_bytes = width * active_rows * ((3 * 4) + 4 + 4 + (3 * 4))
    return output_bytes + strip_working_bytes


def _render_mosaic(
    tiles: list[_TileRecord],
    *,
    max_dimension: int | None = None,
    strip_height: int = MOSAIC_RENDER_STRIP_HEIGHT,
):
    if not tiles:
        return None
    cv2, np = _ensure_cv_numpy()
    min_x = math.floor(min(tile.x for tile in tiles))
    min_y = math.floor(min(tile.y for tile in tiles))
    width, height = _mosaic_dimensions(tiles)
    scale = 1.0
    if max_dimension is not None and max(width, height) > max_dimension:
        scale = max_dimension / max(width, height)
        width = max(1, int(round(width * scale)))
        height = max(1, int(round(height * scale)))
    placements: list[tuple[Any, int, int]] = []
    for tile in tiles:
        tile_image = tile.bgr
        if scale != 1.0:
            tile_image = cv2.resize(
                tile_image,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )
        th, tw = tile_image.shape[:2]
        x = int(round((tile.x - min_x) * scale))
        y = int(round((tile.y - min_y) * scale))
        if min(width, x + tw) <= max(0, x) or min(height, y + th) <= max(0, y):
            continue
        placements.append((tile_image, x, y))

    output = np.zeros((height, width, 3), dtype=np.uint8)
    feather_axes_cache: dict[tuple[int, int], tuple[Any, Any]] = {}
    rows_per_strip = min(height, max(1, int(strip_height)))
    for strip_y in range(0, height, rows_per_strip):
        strip_y2 = min(height, strip_y + rows_per_strip)
        active_height = strip_y2 - strip_y
        canvas = np.zeros((active_height, width, 3), dtype=np.float32)
        weights = np.zeros((active_height, width, 1), dtype=np.float32)
        for tile_image, x, y in placements:
            th, tw = tile_image.shape[:2]
            overlap_x = max(0, x)
            overlap_x2 = min(width, x + tw)
            overlap_y = max(strip_y, y)
            overlap_y2 = min(strip_y2, y + th)
            if overlap_x2 <= overlap_x or overlap_y2 <= overlap_y:
                continue
            tile_x = overlap_x - x
            tile_x2 = overlap_x2 - x
            tile_y = overlap_y - y
            tile_y2 = overlap_y2 - y
            canvas_y = overlap_y - strip_y
            canvas_y2 = overlap_y2 - strip_y
            key = (tw, th)
            if key not in feather_axes_cache:
                feather_axes_cache[key] = _feather_axes(tw, th)
            edge_x, edge_y = feather_axes_cache[key]
            mask = np.minimum(
                edge_y[tile_y:tile_y2, None],
                edge_x[None, tile_x:tile_x2],
            )
            np.multiply(mask, 8.0, out=mask)
            np.clip(mask, 0.12, 1.0, out=mask)
            mask = mask[..., None]
            weighted_tile = tile_image[tile_y:tile_y2, tile_x:tile_x2].astype(np.float32)
            np.multiply(weighted_tile, mask, out=weighted_tile)
            canvas[canvas_y:canvas_y2, overlap_x:overlap_x2] += weighted_tile
            weights[canvas_y:canvas_y2, overlap_x:overlap_x2] += mask
        np.maximum(weights, 1e-6, out=weights)
        np.divide(canvas, weights, out=canvas)
        np.clip(canvas, 0, 255, out=canvas)
        output[strip_y:strip_y2] = canvas
    return output


def _feather_axes(width: int, height: int):
    _, np = _ensure_cv_numpy()
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)
    return np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y)


def _feather_mask(width: int, height: int):
    _, np = _ensure_cv_numpy()
    edge_x, edge_y = _feather_axes(width, height)
    mask = np.minimum.outer(edge_y, edge_x)
    mask = np.clip(mask * 8.0, 0.12, 1.0)
    return mask[..., None]


def log_preview_analysis_perf(title: str, elapsed_ms: float, *, detail: str = "") -> None:
    aggregate_runtime_metric(title, elapsed_ms, detail=detail, interval_s=5.0)
