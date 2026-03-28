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

    def finalize(self) -> FocusStackFinalResult:
        if not self._accumulator.has_frames():
            raise RuntimeError("景深合成未收到有效采样帧。")
        image = self._accumulator.final_image()
        sampled = self._accumulator.sampled_frames
        accepted = self._accumulator.accepted_frames
        metadata = {
            "analysis_mode": "focus_stack",
            "device_id": self._device_id,
            "device_name": self._device_name,
            "sampled_frames": sampled,
            "accepted_frames": accepted,
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
        self._tiles: list[_TileRecord] = []
        self._edges: list[_TileEdge] = []
        self._current_accumulator = FocusAccumulator()
        self._current_origin_small = None
        self._current_predicted_position = (0.0, 0.0)
        self._pending_transition = (0.0, 0.0)
        self._tile_counter = 0
        self._last_message = "等待移动样品台并采样"
        self._tile_shift_threshold_px: float | None = None

    def add_frame(self, image: QImage) -> MapBuildReport:
        frame = _prepare_frame(image)
        self._sampled_frames += 1
        if self._tile_shift_threshold_px is None:
            self._tile_shift_threshold_px = max(72.0, min(frame.bgr.shape[0], frame.bgr.shape[1]) * 0.18)
        if not self._current_accumulator.has_frames():
            self._current_origin_small = frame.small_gray
            self._current_accumulator.add_prepared_frame(frame)
            self._accepted_frames += 1
            self._last_message = "已开始采集第 1 个 tile"
            return self._build_report()

        shift_dx = 0.0
        shift_dy = 0.0
        response = 0.0
        if self._current_origin_small is not None:
            shift_dx, shift_dy, response = _estimate_translation(self._current_origin_small, frame.small_gray)
            shift_dx *= _small_frame_scale(frame.gray.shape, frame.small_gray.shape)
            shift_dy *= _small_frame_scale(frame.gray.shape, frame.small_gray.shape)
        if response > 0.015 and math.hypot(shift_dx, shift_dy) >= float(self._tile_shift_threshold_px or 80.0):
            transition = (shift_dx, shift_dy)
            self._pending_transition = transition
            self._finalize_current_tile()
            if self._tiles:
                last_tile = self._tiles[-1]
                self._current_predicted_position = (last_tile.x + transition[0], last_tile.y + transition[1])
            else:
                self._current_predicted_position = transition
            self._current_origin_small = frame.small_gray
            self._current_accumulator = FocusAccumulator()
            self._current_accumulator.add_prepared_frame(frame)
            self._accepted_frames += 1
            self._last_message = f"已切换到第 {len(self._tiles) + 1} 个 tile"
            return self._build_report()

        accepted = self._current_accumulator.add_prepared_frame(frame)
        if accepted:
            self._accepted_frames += 1
            self._last_message = (
                f"tile {len(self._tiles) + 1} 采样中 | 当前 tile 接受 {self._current_accumulator.accepted_frames} 帧"
            )
        else:
            self._last_message = f"tile {len(self._tiles) + 1} 重复帧已跳过"
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
        return MapBuildReport(
            preview_image=bgr_array_to_qimage(mosaic) if mosaic is not None else QImage(),
            sampled_frames=self._sampled_frames,
            accepted_frames=self._accepted_frames,
            tile_count=len(self._tiles) + (1 if self._current_accumulator.has_frames() else 0),
            message=message,
            low_confidence=bool(warning),
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

    def _refine_edges_for_tile(self, tile: _TileRecord) -> None:
        if len(self._tiles) == 1:
            return
        candidates = self._tiles[:-1][-6:]
        for candidate in candidates:
            predicted_dx = tile.x - candidate.x
            predicted_dy = tile.y - candidate.y
            overlap_ok = _predicted_overlap_ratio(
                candidate.width,
                candidate.height,
                tile.width,
                tile.height,
                predicted_dx,
                predicted_dy,
            ) >= 0.08
            if not overlap_ok and candidate is not self._tiles[-2]:
                continue
            refined_dx, refined_dy, response = _refine_translation(candidate.gray, tile.gray, predicted_dx, predicted_dy)
            if response <= 0.01 and candidate is not self._tiles[-2]:
                continue
            self._edges.append(
                _TileEdge(
                    source_id=candidate.tile_id,
                    target_id=tile.tile_id,
                    dx=refined_dx,
                    dy=refined_dy,
                    weight=max(0.05, float(response)),
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
