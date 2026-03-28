from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import math
from typing import Any

from PySide6.QtGui import QImage

from fdm.runtime_logging import append_runtime_log


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
    transition_predicted_dx: float = 0.0
    transition_predicted_dy: float = 0.0
    transition_refined_dx: float = 0.0
    transition_refined_dy: float = 0.0
    transition_method: str = ""
    transition_ncc: float = 0.0
    transition_delta_from_prediction: float = 0.0


@dataclass(slots=True)
class MapBuildFinalResult:
    image: QImage
    sampled_frames: int
    accepted_frames: int
    tile_count: int
    metadata: dict[str, Any]


@dataclass(slots=True)
class _PreparedFrame:
    bgr: Any
    gray: Any
    focus_map: Any
    small_gray: Any
    sharpness: float


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
class _TransitionTracker:
    start_position: tuple[float, float]
    cumulative_dx: float = 0.0
    cumulative_dy: float = 0.0
    best_stable_frames: list[_PreparedFrame] = field(default_factory=list)

    def observe_step(self, dx: float, dy: float, *, response: float, max_step: float) -> None:
        if response < 0.2 or math.hypot(dx, dy) > max_step:
            return
        self.cumulative_dx += dx
        self.cumulative_dy += dy

    def predicted_shift(self) -> tuple[float, float]:
        return self.cumulative_dx, self.cumulative_dy

    def predicted_position(self) -> tuple[float, float]:
        return (self.start_position[0] + self.cumulative_dx, self.start_position[1] + self.cumulative_dy)

    def note_stable_frame(self, frame: _PreparedFrame) -> None:
        self.best_stable_frames.append(frame)
        self.best_stable_frames.sort(key=lambda item: item.sharpness, reverse=True)
        if len(self.best_stable_frames) > 4:
            self.best_stable_frames = self.best_stable_frames[:4]


@dataclass(slots=True)
class _RegistrationResult:
    accepted: bool
    dx: float
    dy: float
    response: float
    ncc: float
    method: str
    delta_from_prediction: float
    inlier_count: int = 0


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

    def preview_image(self) -> QImage:
        if not self._records:
            return QImage()
        if len(self._records) == 1:
            return bgr_array_to_qimage(self._records[0].bgr)
        blended = _focus_stack_fast([record.bgr for record in self._records], [record.focus_map for record in self._records])
        return bgr_array_to_qimage(blended)

    def final_image(self) -> QImage:
        if not self._records:
            return QImage()
        if len(self._records) == 1:
            return bgr_array_to_qimage(self._records[0].bgr)
        blended = _focus_stack_multiscale(
            [record.bgr for record in self._records],
            [record.focus_map for record in self._records],
        )
        return bgr_array_to_qimage(blended)

    def latest_sharpness(self) -> float:
        if not self._records:
            return 0.0
        return float(self._records[-1].sharpness)


class FocusStackAnalyzer:
    def __init__(self, *, device_id: str, device_name: str) -> None:
        self._device_id = device_id
        self._device_name = device_name
        self._accumulator = FocusAccumulator()

    def add_frame(self, image: QImage) -> FocusStackReport:
        accepted = self._accumulator.add_qimage(image)
        preview = self._accumulator.preview_image()
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

    def finalize(self, *, post_sharpen: bool = False) -> FocusStackFinalResult:
        if not self._accumulator.has_frames():
            raise RuntimeError("景深合成未收到有效采样帧。")
        image = self._accumulator.final_image()
        if post_sharpen and not image.isNull():
            image = _apply_post_sharpen(image)
        sampled = self._accumulator.sampled_frames
        accepted = self._accumulator.accepted_frames
        metadata = {
            "analysis_mode": "focus_stack",
            "device_id": self._device_id,
            "device_name": self._device_name,
            "sampled_frames": sampled,
            "accepted_frames": accepted,
            "post_sharpen": bool(post_sharpen),
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
        self._sampled_frames = 0
        self._accepted_frames = 0
        self._rejected_moving_frames = 0
        self._rejected_low_confidence_frames = 0
        self._stable_accept_count = 0
        self._tiles: list[_TileRecord] = []
        self._edges: list[_TileEdge] = []
        self._current_accumulator = FocusAccumulator()
        self._current_origin_small = None
        self._current_predicted_position = (0.0, 0.0)
        self._pending_transition = (0.0, 0.0)
        self._tile_counter = 0
        self._previous_frame: _PreparedFrame | None = None
        self._transition_tracker: _TransitionTracker | None = None
        self._transition_pending = False
        self._is_stable = False
        self._stable_streak = 0
        self._unstable_streak = 0
        self._stable_window: list[_PreparedFrame] = []
        self._stable_required = 3
        self._last_message = "等待移动样品台并采样"
        self._tile_shift_threshold_px: float | None = None
        self._stable_step_threshold_px: float | None = None
        self._tile_freeze_threshold_px: float | None = None
        self._stable_response_threshold = 0.015
        self._resume_origin_threshold_px: float | None = None
        self._transition_search_radius_px: float | None = None
        self._last_translation_px = 0.0
        self._last_response = 0.0
        self._last_quality_score = 0.0
        self._last_transition_predicted_dx = 0.0
        self._last_transition_predicted_dy = 0.0
        self._last_transition_refined_dx = 0.0
        self._last_transition_refined_dy = 0.0
        self._last_transition_method = ""
        self._last_transition_ncc = 0.0
        self._last_transition_delta = 0.0

    def add_frame(self, image: QImage) -> MapBuildReport:
        frame = _prepare_frame(image)
        self._sampled_frames += 1
        if self._tile_shift_threshold_px is None:
            self._tile_shift_threshold_px = max(20.0, min(frame.bgr.shape[0], frame.bgr.shape[1]) * 0.06)
        if self._stable_step_threshold_px is None:
            self._stable_step_threshold_px = max(2.0, min(frame.bgr.shape[0], frame.bgr.shape[1]) * 0.004)
        if self._tile_freeze_threshold_px is None:
            self._tile_freeze_threshold_px = max(4.0, min(frame.bgr.shape[0], frame.bgr.shape[1]) * 0.01)
        if self._resume_origin_threshold_px is None:
            self._resume_origin_threshold_px = max(3.0, float(self._tile_freeze_threshold_px or 4.0) * 0.75)
        if self._transition_search_radius_px is None:
            self._transition_search_radius_px = max(18.0, float(self._tile_shift_threshold_px or 20.0) * 0.75)
        self._last_quality_score = frame.sharpness
        self._last_transition_predicted_dx = 0.0
        self._last_transition_predicted_dy = 0.0
        self._last_transition_refined_dx = 0.0
        self._last_transition_refined_dy = 0.0
        self._last_transition_method = ""
        self._last_transition_ncc = 0.0
        self._last_transition_delta = 0.0
        if self._previous_frame is None:
            self._previous_frame = frame
            self._stable_window = [frame]
            self._stable_streak = 1
            self._unstable_streak = 0
            self._last_translation_px = 0.0
            self._last_response = 1.0
            self._last_message = self._settling_message()
            return self._build_report()

        step_dx, step_dy, step_response = _estimate_translation(self._previous_frame.small_gray, frame.small_gray)
        step_scale = _small_frame_scale(frame.gray.shape, frame.small_gray.shape)
        step_dx *= step_scale
        step_dy *= step_scale
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
        origin_response = 0.0
        origin_translation = 0.0
        if self._current_origin_small is not None:
            origin_dx, origin_dy, origin_response = _estimate_translation(self._current_origin_small, frame.small_gray)
            origin_dx *= step_scale
            origin_dy *= step_scale
            origin_translation = math.hypot(origin_dx, origin_dy)

        if self._current_accumulator.has_frames() and origin_translation >= float(self._tile_freeze_threshold_px or 6.0):
            if not self._transition_pending:
                self._transition_pending = True
                self._transition_tracker = _TransitionTracker(start_position=self._current_predicted_position)
                self._is_stable = False
                self._unstable_streak = 0
                if step_translation <= float(self._stable_step_threshold_px or 2.0) and step_response >= self._stable_response_threshold:
                    self._stable_streak = 1
                    self._stable_window = [frame]
                else:
                    self._stable_streak = 0
                    self._stable_window.clear()
            if self._transition_tracker is not None:
                self._transition_tracker.observe_step(
                    step_dx,
                    step_dy,
                    response=step_response,
                    max_step=max(float(self._transition_search_radius_px or 18.0) * 2.0, float(self._tile_shift_threshold_px or 20.0) * 1.5),
                )
            self._pending_transition = (origin_dx, origin_dy)

        if not self._current_accumulator.has_frames():
            if self._is_stable and stable_anchor is not None:
                self._current_origin_small = stable_anchor.small_gray
                self._current_predicted_position = (0.0, 0.0)
                self._accept_prepared_frame(stable_anchor)
                self._last_message = self._sampling_message()
            else:
                self._rejected_moving_frames += 1
                self._last_message = self._motion_wait_message()
            return self._build_report()

        if self._transition_pending:
            if not self._is_stable:
                self._rejected_moving_frames += 1
                self._last_message = "检测到新位置，等待稳定"
                return self._build_report()

            candidate = stable_anchor or self._best_stable_frame()
            if candidate is None:
                self._last_message = "检测到新位置，等待稳定"
                return self._build_report()
            if self._transition_tracker is not None:
                self._transition_tracker.note_stable_frame(candidate)
                predicted_dx, predicted_dy = self._transition_tracker.predicted_shift()
            else:
                predicted_dx, predicted_dy = self._pending_transition
            self._last_transition_predicted_dx = predicted_dx
            self._last_transition_predicted_dy = predicted_dy

            current_reference = self._current_tile_reference_frame()
            registration = self._register_transition_candidate(
                current_reference,
                candidate,
                predicted_dx=predicted_dx,
                predicted_dy=predicted_dy,
            )
            self._last_transition_refined_dx = registration.dx
            self._last_transition_refined_dy = registration.dy
            self._last_transition_method = registration.method
            self._last_transition_ncc = registration.ncc
            self._last_transition_delta = registration.delta_from_prediction

            candidate_translation = math.hypot(registration.dx, registration.dy)
            if candidate_translation <= float(self._resume_origin_threshold_px or 4.0):
                self._transition_pending = False
                self._transition_tracker = None
                if self._accept_prepared_frame(candidate):
                    self._last_message = self._sampling_message()
                else:
                    self._last_message = f"{self._sampling_message()} | 当前帧与上一帧接近"
                return self._build_report()
            if not registration.accepted:
                self._rejected_low_confidence_frames += 1
                self._last_message = "重叠纹理不足或匹配不可靠，未创建新 tile"
                return self._build_report()
            transition = (registration.dx, registration.dy)
            self._pending_transition = transition
            self._finalize_current_tile()
            if self._tiles:
                last_tile = self._tiles[-1]
                self._current_predicted_position = (last_tile.x + transition[0], last_tile.y + transition[1])
            else:
                self._current_predicted_position = transition
            self._current_origin_small = candidate.small_gray
            self._transition_pending = False
            self._transition_tracker = None
            if self._accept_prepared_frame(candidate):
                self._last_message = self._sampling_message()
            else:
                self._last_message = f"{self._sampling_message()} | 当前帧与上一帧接近"
            return self._build_report()

        if not self._is_stable:
            self._rejected_moving_frames += 1
            self._last_message = self._motion_wait_message()
            return self._build_report()

        candidate = stable_anchor if newly_stable and stable_anchor is not None else frame
        if self._accept_prepared_frame(candidate):
            self._last_message = self._sampling_message()
        else:
            self._last_message = f"{self._sampling_message()} | 当前帧与上一帧接近"
        return self._build_report()

    def finalize(self) -> MapBuildFinalResult:
        self._finalize_current_tile()
        if not self._tiles:
            raise RuntimeError("地图构建未生成有效 tile。")
        image = _render_mosaic(self._tiles)
        metadata = {
            "analysis_mode": "map_build",
            "device_id": self._device_id,
            "device_name": self._device_name,
            "sampled_frames": self._sampled_frames,
            "accepted_frames": self._accepted_frames,
            "tile_count": len(self._tiles),
            "rejected_moving_frames": self._rejected_moving_frames,
            "rejected_low_confidence_frames": self._rejected_low_confidence_frames,
            "stable_accept_count": self._stable_accept_count,
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
        preview_tiles = list(self._tiles)
        if self._current_accumulator.has_frames():
            current_preview = self._current_accumulator.preview_image()
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
            preview_image=bgr_array_to_qimage(mosaic) if mosaic is not None else QImage(),
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
            transition_predicted_dx=self._last_transition_predicted_dx,
            transition_predicted_dy=self._last_transition_predicted_dy,
            transition_refined_dx=self._last_transition_refined_dx,
            transition_refined_dy=self._last_transition_refined_dy,
            transition_method=self._last_transition_method,
            transition_ncc=self._last_transition_ncc,
            transition_delta_from_prediction=self._last_transition_delta,
        )

    def _finalize_current_tile(self) -> None:
        if not self._current_accumulator.has_frames():
            return
        tile_image = self._current_accumulator.final_image()
        if tile_image.isNull():
            return
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
        self._refine_edges_for_tile(tile)
        self._optimize_tile_positions()
        self._current_accumulator = FocusAccumulator()
        self._current_origin_small = None
        self._pending_transition = (0.0, 0.0)
        self._transition_tracker = None

    def _current_tile_reference_frame(self) -> _PreparedFrame:
        current_preview = self._current_accumulator.preview_image()
        if current_preview.isNull():
            raise RuntimeError("当前 tile 为空，无法继续地图构建。")
        return _prepare_frame(current_preview)

    def _register_transition_candidate(
        self,
        reference: _PreparedFrame,
        candidate: _PreparedFrame,
        *,
        predicted_dx: float,
        predicted_dy: float,
    ) -> _RegistrationResult:
        search_radius = max(28.0, float(self._transition_search_radius_px or 18.0))
        overlap_min = 0.10
        phase_min = max(0.015, self._stable_response_threshold)
        ncc_min = 0.16
        delta_limit = max(search_radius * 1.6, 20.0)
        phase_candidates: list[tuple[float, float, float, float, float, float]] = []
        seen: set[tuple[int, int]] = set()
        for seed_dx, seed_dy in self._registration_seed_candidates(predicted_dx, predicted_dy):
            rounded = (int(round(seed_dx)), int(round(seed_dy)))
            if rounded in seen:
                continue
            seen.add(rounded)
            dx, dy, response, ncc = _search_local_translation_near_prediction(
                reference.gray,
                candidate.gray,
                seed_dx,
                seed_dy,
                search_radius=search_radius,
            )
            delta = math.hypot(dx - seed_dx, dy - seed_dy)
            overlap_ratio = _predicted_overlap_ratio(
                reference.bgr.shape[1],
                reference.bgr.shape[0],
                candidate.bgr.shape[1],
                candidate.bgr.shape[0],
                dx,
                dy,
            )
            phase_candidates.append((dx, dy, response, ncc, delta, overlap_ratio))
        if not phase_candidates:
            phase_candidates.append((predicted_dx, predicted_dy, 0.0, -1.0, 0.0, 0.0))
        dx, dy, response, ncc, delta, overlap_ratio = max(
            phase_candidates,
            key=lambda item: (item[3], item[2], -item[4]),
        )
        if response >= phase_min and ncc >= ncc_min and overlap_ratio >= overlap_min and delta <= delta_limit:
            return _RegistrationResult(
                accepted=True,
                dx=dx,
                dy=dy,
                response=response,
                ncc=ncc,
                method="phase_local",
                delta_from_prediction=delta,
            )
        orb_dx, orb_dy, orb_ncc, inliers = _estimate_translation_orb(
            reference.gray,
            candidate.gray,
            predicted_dx,
            predicted_dy,
        )
        orb_delta = math.hypot(orb_dx - predicted_dx, orb_dy - predicted_dy)
        orb_overlap = _predicted_overlap_ratio(
            reference.bgr.shape[1],
            reference.bgr.shape[0],
            candidate.bgr.shape[1],
            candidate.bgr.shape[0],
            orb_dx,
            orb_dy,
        )
        if inliers >= 5 and orb_ncc >= max(0.12, ncc_min - 0.04) and orb_overlap >= overlap_min and orb_delta <= delta_limit:
            return _RegistrationResult(
                accepted=True,
                dx=orb_dx,
                dy=orb_dy,
                response=response,
                ncc=orb_ncc,
                method="orb_fallback",
                delta_from_prediction=orb_delta,
                inlier_count=inliers,
            )
        return _RegistrationResult(
            accepted=False,
            dx=dx,
            dy=dy,
            response=response,
            ncc=ncc,
            method="rejected",
            delta_from_prediction=delta,
            inlier_count=inliers,
        )

    def _refine_edges_for_tile(self, tile: _TileRecord) -> None:
        if len(self._tiles) == 1:
            return
        candidate = self._tiles[-2]
        predicted_dx = tile.x - candidate.x
        predicted_dy = tile.y - candidate.y
        registration = self._register_transition_candidate(
            _PreparedFrame(
                bgr=candidate.bgr,
                gray=candidate.gray,
                focus_map=None,
                small_gray=candidate.gray,
                sharpness=0.0,
            ),
            _PreparedFrame(
                bgr=tile.bgr,
                gray=tile.gray,
                focus_map=None,
                small_gray=tile.gray,
                sharpness=0.0,
            ),
            predicted_dx=predicted_dx,
            predicted_dy=predicted_dy,
        )
        if not registration.accepted:
            return
        self._edges.append(
            _TileEdge(
                source_id=candidate.tile_id,
                target_id=tile.tile_id,
                dx=registration.dx,
                dy=registration.dy,
                weight=max(0.10, float(min(1.0, registration.response + registration.ncc * 0.35))),
            )
        )

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
            return "重叠不足，当前地图使用顺序拼接"
        if self._edges and self._edges[-1].weight <= 0.08:
            return "最近 tile 匹配置信度较低"
        return ""

    def _motion_state(self) -> str:
        if self._transition_pending:
            return "transition_pending"
        if self._is_stable:
            return "stable"
        if self._stable_streak > 0:
            return "settling"
        return "moving"

    def _settling_message(self) -> str:
        return f"静止确认中 {min(self._stable_streak, self._stable_required)}/{self._stable_required}"

    def _sampling_message(self) -> str:
        return f"已静止，正在采样 tile {len(self._tiles) + 1}"

    def _motion_wait_message(self) -> str:
        if self._transition_pending:
            return "检测到新位置，等待稳定"
        if self._stable_streak > 0:
            return self._settling_message()
        return "运动中，暂停入图"

    def _registration_seed_candidates(self, predicted_dx: float, predicted_dy: float) -> list[tuple[float, float]]:
        candidates: list[tuple[float, float]] = [(predicted_dx, predicted_dy)]
        if self._pending_transition != (0.0, 0.0):
            candidates.append(self._pending_transition)
        if self._transition_tracker is not None:
            tracker_dx, tracker_dy = self._transition_tracker.predicted_shift()
            candidates.append((tracker_dx, tracker_dy))
        candidates.append((predicted_dx * 0.5, predicted_dy * 0.5))
        candidates.append((0.0, 0.0))
        return candidates

    def _best_stable_frame(self) -> _PreparedFrame | None:
        if not self._stable_window:
            return None
        return max(self._stable_window, key=lambda frame: frame.sharpness)

    def _update_stability_gate(
        self,
        frame: _PreparedFrame,
        *,
        translation_px: float,
        response: float,
        allow_soft_stable: bool,
    ) -> tuple[bool, _PreparedFrame | None]:
        stable_threshold = float(self._stable_step_threshold_px or 2.0)
        stationary = translation_px <= stable_threshold and response >= self._stable_response_threshold
        soft_stationary = allow_soft_stable and translation_px <= stable_threshold * 0.55
        if stationary or soft_stationary:
            self._stable_streak += 1
            self._unstable_streak = 0
            self._stable_window.append(frame)
            max_window = max(4, self._stable_required + 1)
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

    def _accept_prepared_frame(self, frame: _PreparedFrame) -> bool:
        accepted = self._current_accumulator.add_prepared_frame(frame)
        if accepted:
            self._accepted_frames += 1
            self._stable_accept_count += 1
        return accepted


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


def _focus_stack_multiscale(images: list, focus_maps: list):
    cv2, np = _ensure_cv_numpy()
    image_stack = np.stack([image.astype(np.float32, copy=False) for image in images], axis=0)
    weight_sum = np.zeros(focus_maps[0].shape, dtype=np.float32)
    total_weights = np.zeros((len(focus_maps),) + focus_maps[0].shape, dtype=np.float32)
    sigmas = (1.0, 2.5, 5.0)
    for sigma in sigmas:
        smoothed = np.stack([cv2.GaussianBlur(focus, (0, 0), sigmaX=sigma, sigmaY=sigma) for focus in focus_maps], axis=0)
        smoothed += 1e-6
        smoothed /= smoothed.sum(axis=0, keepdims=True)
        total_weights += smoothed
        weight_sum += smoothed.sum(axis=0)
    total_weights /= max(1.0, float(len(sigmas)))
    total_weights /= np.clip(total_weights.sum(axis=0, keepdims=True), 1e-6, None)
    fused = np.sum(image_stack * total_weights[..., None], axis=0)
    return np.clip(fused, 0, 255).astype(np.uint8)


def _apply_post_sharpen(image: QImage) -> QImage:
    cv2, np = _ensure_cv_numpy()
    bgr = qimage_to_bgr_array(image)
    blurred = cv2.GaussianBlur(bgr, (0, 0), sigmaX=1.1, sigmaY=1.1)
    sharpened = cv2.addWeighted(bgr, 1.35, blurred, -0.35, 0.0)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8, copy=False)
    return bgr_array_to_qimage(sharpened)


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


def _crop_overlap(gray_a, gray_b, dx: float, dy: float):
    _, np = _ensure_cv_numpy()
    dx_i = int(round(dx))
    dy_i = int(round(dy))
    x1_a = max(0, dx_i)
    y1_a = max(0, dy_i)
    x1_b = max(0, -dx_i)
    y1_b = max(0, -dy_i)
    overlap_w = min(gray_a.shape[1] - x1_a, gray_b.shape[1] - x1_b)
    overlap_h = min(gray_a.shape[0] - y1_a, gray_b.shape[0] - y1_b)
    if overlap_w < 64 or overlap_h < 64:
        return None, None
    crop_a = gray_a[y1_a : y1_a + overlap_h, x1_a : x1_a + overlap_w]
    crop_b = gray_b[y1_b : y1_b + overlap_h, x1_b : x1_b + overlap_w]
    if crop_a.size == 0 or crop_b.size == 0:
        return None, None
    return crop_a, crop_b


def _refine_translation(gray_a, gray_b, predicted_dx: float, predicted_dy: float) -> tuple[float, float, float]:
    crop_a, crop_b = _crop_overlap(gray_a, gray_b, predicted_dx, predicted_dy)
    if crop_a is None or crop_b is None:
        return predicted_dx, predicted_dy, 0.0
    residual_dx, residual_dy, response = _estimate_translation(crop_a, crop_b)
    return predicted_dx + residual_dx, predicted_dy + residual_dy, response


def _normalized_cross_correlation(gray_a, gray_b, dx: float, dy: float) -> float:
    _, np = _ensure_cv_numpy()
    crop_a, crop_b = _crop_overlap(gray_a, gray_b, dx, dy)
    if crop_a is None or crop_b is None:
        return -1.0
    a = crop_a.astype(np.float32, copy=False)
    b = crop_b.astype(np.float32, copy=False)
    a -= float(a.mean())
    b -= float(b.mean())
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-6:
        return -1.0
    return float(np.sum(a * b) / denom)


def _search_local_translation_near_prediction(gray_a, gray_b, predicted_dx: float, predicted_dy: float, *, search_radius: float) -> tuple[float, float, float, float]:
    cv2, _ = _ensure_cv_numpy()
    coarse_best_dx = predicted_dx
    coarse_best_dy = predicted_dy
    coarse_best_ncc = -1.0
    radius = max(4, int(round(search_radius)))
    coarse_step = max(2, radius // 4)
    for dy in range(-radius, radius + 1, coarse_step):
        for dx in range(-radius, radius + 1, coarse_step):
            trial_dx = predicted_dx + dx
            trial_dy = predicted_dy + dy
            ncc = _normalized_cross_correlation(gray_a, gray_b, trial_dx, trial_dy)
            if ncc > coarse_best_ncc:
                coarse_best_ncc = ncc
                coarse_best_dx = trial_dx
                coarse_best_dy = trial_dy
    fine_best_dx = coarse_best_dx
    fine_best_dy = coarse_best_dy
    fine_best_ncc = coarse_best_ncc
    for dy in range(-max(2, coarse_step), max(2, coarse_step) + 1):
        for dx in range(-max(2, coarse_step), max(2, coarse_step) + 1):
            trial_dx = coarse_best_dx + dx
            trial_dy = coarse_best_dy + dy
            ncc = _normalized_cross_correlation(gray_a, gray_b, trial_dx, trial_dy)
            if ncc > fine_best_ncc:
                fine_best_ncc = ncc
                fine_best_dx = trial_dx
                fine_best_dy = trial_dy
    crop_a, crop_b = _crop_overlap(gray_a, gray_b, fine_best_dx, fine_best_dy)
    if crop_a is None or crop_b is None:
        return fine_best_dx, fine_best_dy, 0.0, fine_best_ncc
    residual_dx, residual_dy, response = _estimate_translation(crop_a, crop_b)
    if abs(residual_dx) > 6.0 or abs(residual_dy) > 6.0:
        residual_dx = 0.0
        residual_dy = 0.0
    refined_dx = fine_best_dx + residual_dx
    refined_dy = fine_best_dy + residual_dy
    refined_ncc = _normalized_cross_correlation(gray_a, gray_b, refined_dx, refined_dy)
    if refined_ncc < fine_best_ncc:
        refined_dx = fine_best_dx
        refined_dy = fine_best_dy
        refined_ncc = fine_best_ncc
    return refined_dx, refined_dy, float(response), refined_ncc


def _estimate_translation_orb(gray_a, gray_b, predicted_dx: float, predicted_dy: float) -> tuple[float, float, float, int]:
    cv2, np = _ensure_cv_numpy()
    crop_a, crop_b = _crop_overlap(gray_a, gray_b, predicted_dx, predicted_dy)
    if crop_a is None or crop_b is None:
        return predicted_dx, predicted_dy, -1.0, 0
    orb = cv2.ORB_create(nfeatures=800, fastThreshold=10)
    keypoints_a, descriptors_a = orb.detectAndCompute(crop_a, None)
    keypoints_b, descriptors_b = orb.detectAndCompute(crop_b, None)
    if descriptors_a is None or descriptors_b is None or len(keypoints_a) < 8 or len(keypoints_b) < 8:
        return predicted_dx, predicted_dy, -1.0, 0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(descriptors_a, descriptors_b), key=lambda item: item.distance)
    if len(matches) < 8:
        return predicted_dx, predicted_dy, -1.0, len(matches)
    matches = matches[: min(120, len(matches))]
    points_a = np.float32([keypoints_a[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
    points_b = np.float32([keypoints_b[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)
    matrix, inliers = cv2.estimateAffinePartial2D(
        points_b,
        points_a,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=2000,
        confidence=0.99,
    )
    if matrix is None or inliers is None:
        return predicted_dx, predicted_dy, -1.0, 0
    residual_dx = float(matrix[0, 2])
    residual_dy = float(matrix[1, 2])
    dx = predicted_dx + residual_dx
    dy = predicted_dy + residual_dy
    ncc = _normalized_cross_correlation(gray_a, gray_b, dx, dy)
    return dx, dy, ncc, int(inliers.sum())


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
