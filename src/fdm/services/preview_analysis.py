from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from time import perf_counter
from typing import Any

from PySide6.QtGui import QImage

from fdm.runtime_logging import append_runtime_log
from fdm.settings import FocusStackProfile

MAP_BUILD_ANALYSIS_INTERVAL_MS = 90
MAP_BUILD_STABLE_REQUIRED_FRAMES = 2
MAP_BUILD_MAX_TILE_FRAMES = 3
MAP_BUILD_PREVIEW_REFRESH_INTERVAL_MS = 250


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


def map_build_parameter_defaults() -> dict[str, Any]:
    return {
        "analysis_interval_ms": MAP_BUILD_ANALYSIS_INTERVAL_MS,
        "stable_required_frames": MAP_BUILD_STABLE_REQUIRED_FRAMES,
        "max_tile_frames": MAP_BUILD_MAX_TILE_FRAMES,
        "preview_refresh_interval_ms": MAP_BUILD_PREVIEW_REFRESH_INTERVAL_MS,
        "registration_thresholds": _MapRegistrationConfig().as_metadata(),
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
    def __init__(self) -> None:
        self._records: list[_PreparedFrame] = []
        self.sampled_frames = 0
        self.accepted_frames = 0

    def has_frames(self) -> bool:
        return bool(self._records)

    def add_qimage(self, image: QImage) -> bool:
        frame = _prepare_frame(image)
        return self.add_prepared_frame(frame)

    def add_prepared_frame(self, frame: _PreparedFrame) -> bool:
        self.sampled_frames += 1
        if self._records and _is_duplicate_frame(frame, self._records[-1]):
            return False
        self._records.append(frame)
        self.accepted_frames += 1
        return True

    def render_image(self, render_config: FocusStackRenderConfig | None = None) -> QImage:
        if not self._records:
            return QImage()
        config = (render_config or FocusStackRenderConfig()).normalized_copy()
        if len(self._records) == 1:
            blended = self._records[0].bgr.copy()
        else:
            blended = _focus_stack_render(
                [record.bgr for record in self._records],
                [record.focus_map for record in self._records],
                config,
            )
        blended = _apply_sharpen_strength(blended, config.sharpen_strength)
        return bgr_array_to_qimage(blended)

    def preview_image(self, render_config: FocusStackRenderConfig | None = None) -> QImage:
        return self.render_image(render_config)

    def final_image(self, render_config: FocusStackRenderConfig | None = None) -> QImage:
        return self.render_image(render_config)

    def latest_sharpness(self) -> float:
        if not self._records:
            return 0.0
        return float(self._records[-1].sharpness)


class FocusStackAnalyzer:
    def __init__(
        self,
        *,
        device_id: str,
        device_name: str,
        render_config: FocusStackRenderConfig | None = None,
    ) -> None:
        self._device_id = device_id
        self._device_name = device_name
        self._accumulator = FocusAccumulator()
        self._render_config = (render_config or FocusStackRenderConfig()).normalized_copy()

    def add_frame(self, image: QImage) -> FocusStackReport:
        accepted = self._accumulator.add_qimage(image)
        preview = self._accumulator.preview_image(self._render_config)
        sampled = self._accumulator.sampled_frames
        accepted_count = self._accumulator.accepted_frames
        message = f"采样 {sampled} 帧 | 接受 {accepted_count} 帧"
        if not accepted:
            message += " | 重复帧已跳过"
        return FocusStackReport(
            preview_image=preview,
            sampled_frames=sampled,
            accepted_frames=accepted_count,
            message=message,
            low_confidence=accepted_count < 2,
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
    def __init__(self, *, device_id: str, device_name: str) -> None:
        self._device_id = device_id
        self._device_name = device_name
        self._render_config = FocusStackRenderConfig(
            profile=FocusStackProfile.BALANCED,
            sharpen_strength=0,
        )
        self._registration_config = _MapRegistrationConfig()
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
        self._current_accumulator = FocusAccumulator()
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
        light_started = perf_counter()
        frame = _prepare_map_motion_frame(image)
        self._last_perf_metrics["light_motion_prep_ms"] = (perf_counter() - light_started) * 1000.0
        self._sampled_frames += 1
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
                self._accept_motion_frame(stable_anchor)
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
        else:
            self._last_message = f"{self._sampling_message()} | 当前帧与上一帧接近，已跳过"
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
        self._tile_counter += 1
        self._tiles.append(tile)
        if self._pending_current_edge is not None and self._pending_current_edge.target_id == tile.tile_id:
            self._edges.append(self._pending_current_edge)
            self._pending_current_edge = None
        self._optimize_tile_positions()
        self._current_accumulator = FocusAccumulator()
        self._current_origin_small = None
        self._mark_preview_dirty()
        return tile

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
        prepared_candidates = [self._promote_motion_frame(candidate) for candidate in candidate_frames]
        candidate_bgr = _fuse_prepared_frames(prepared_candidates, self._render_config)
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
        self._current_accumulator = FocusAccumulator()
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
        return self._accept_prepared_frame(self._promote_motion_frame(frame))

    def _accept_prepared_frame(self, frame: _PreparedFrame) -> str:
        if self._current_accumulator.accepted_frames >= MAP_BUILD_MAX_TILE_FRAMES:
            self._skipped_tile_frames += 1
            return "full"
        accepted = self._current_accumulator.add_prepared_frame(frame)
        if accepted:
            self._accepted_frames += 1
            self._stable_accept_count += 1
            self._mark_preview_dirty()
            return "accepted"
        return "duplicate"

    def _promote_motion_frame(self, frame: _MapMotionFrame) -> _PreparedFrame:
        if frame.prepared is None:
            started = perf_counter()
            frame.prepared = _prepare_frame(frame.image)
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


def _fuse_prepared_frames(frames: list[_PreparedFrame], render_config: FocusStackRenderConfig) -> Any | None:
    if not frames:
        return None
    accumulator = FocusAccumulator()
    for frame in frames:
        accumulator.add_prepared_frame(frame)
    if not accumulator.has_frames():
        accumulator.add_prepared_frame(frames[-1])
    image = accumulator.final_image(render_config)
    if image.isNull():
        return None
    return qimage_to_bgr_array(image)


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


def _is_duplicate_frame(frame: _PreparedFrame, previous: _PreparedFrame) -> bool:
    _, np = _ensure_cv_numpy()
    current = frame.small_gray.astype(np.float32, copy=False)
    last = previous.small_gray.astype(np.float32, copy=False)
    diff = float(np.mean(np.abs(current - last)))
    sharpness_delta = abs(frame.sharpness - previous.sharpness) / max(previous.sharpness, 1.0)
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


def _render_mosaic(tiles: list[_TileRecord], *, max_dimension: int | None = None):
    if not tiles:
        return None
    cv2, np = _ensure_cv_numpy()
    min_x = math.floor(min(tile.x for tile in tiles))
    min_y = math.floor(min(tile.y for tile in tiles))
    max_x = math.ceil(max(tile.x + tile.width for tile in tiles))
    max_y = math.ceil(max(tile.y + tile.height for tile in tiles))
    width = max(1, max_x - min_x)
    height = max(1, max_y - min_y)
    scale = 1.0
    if max_dimension is not None and max(width, height) > max_dimension:
        scale = max_dimension / max(width, height)
        width = max(1, int(round(width * scale)))
        height = max(1, int(round(height * scale)))
    canvas = np.zeros((height, width, 3), dtype=np.float32)
    weights = np.zeros((height, width, 1), dtype=np.float32)
    feather_cache: dict[tuple[int, int], Any] = {}
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
        x2 = min(width, x + tw)
        y2 = min(height, y + th)
        if x2 <= x or y2 <= y:
            continue
        tile_crop = tile_image[: y2 - y, : x2 - x].astype(np.float32, copy=False)
        key = (tile_crop.shape[1], tile_crop.shape[0])
        if key not in feather_cache:
            feather_cache[key] = _feather_mask(tile_crop.shape[1], tile_crop.shape[0])
        mask = feather_cache[key][: y2 - y, : x2 - x, :]
        canvas[y:y2, x:x2] += tile_crop * mask
        weights[y:y2, x:x2] += mask
    weights = np.clip(weights, 1e-6, None)
    return np.clip(canvas / weights, 0, 255).astype(np.uint8)


def _feather_mask(width: int, height: int):
    _, np = _ensure_cv_numpy()
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)
    edge_x = np.minimum(x, 1.0 - x)
    edge_y = np.minimum(y, 1.0 - y)
    mask = np.minimum.outer(edge_y, edge_x)
    mask = np.clip(mask * 8.0, 0.12, 1.0)
    return mask[..., None]


def log_preview_analysis_perf(title: str, elapsed_ms: float, *, detail: str = "") -> None:
    message = f"elapsed_ms={elapsed_ms:.2f}"
    if detail:
        message = f"{message}, {detail}"
    append_runtime_log(title, message)
