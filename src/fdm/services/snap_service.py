from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING, Any

from fdm.geometry import Line, Point, clamp, direction, line_length
from fdm.raster import RasterImage

if TYPE_CHECKING:
    from PySide6.QtGui import QImage

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - optional at import time
    np = None  # type: ignore[assignment]


@dataclass(slots=True)
class SnapResult:
    status: str
    original_line: Line
    snapped_line: Line | None
    diameter_px: float | None
    confidence: float
    debug_payload: dict[str, Any] = field(default_factory=dict)


class SnapService:
    def __init__(
        self,
        *,
        sample_step_px: float = 0.5,
        profile_half_width_px: int = 2,
        confidence_review_threshold: float = 0.45,
    ) -> None:
        self.sample_step_px = sample_step_px
        self.profile_half_width_px = profile_half_width_px
        self.confidence_review_threshold = confidence_review_threshold
        self._gaussian_kernel = (1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0)

    def snap_measurement(self, image: "QImage | RasterImage | np.ndarray[Any, Any]", line: Line) -> SnapResult:
        self._require_numpy()
        if line_length(line) < 3.0:
            return SnapResult(
                status="line_too_short",
                original_line=line,
                snapped_line=None,
                diameter_px=None,
                confidence=0.0,
                debug_payload={"reason": "Input line is too short."},
            )

        grayscale = self._to_grayscale_array(image)
        if grayscale.size == 0:
            return SnapResult(
                status="profile_too_flat",
                original_line=line,
                snapped_line=None,
                diameter_px=None,
                confidence=0.0,
                debug_payload={"reason": "Image is empty."},
            )

        profile_data = self._extract_profile(grayscale, line)
        if profile_data is None:
            return SnapResult(
                status="profile_too_flat",
                original_line=line,
                snapped_line=None,
                diameter_px=None,
                confidence=0.0,
                debug_payload={"reason": "Unable to extract a stable profile."},
            )

        positions, profile, axis, image_size = profile_data
        smoothed = self._smooth_profile(profile)
        gradient = np.gradient(smoothed, positions)
        max_abs_gradient = float(np.max(np.abs(gradient))) if gradient.size else 0.0
        contrast = self._center_edge_contrast(smoothed)
        polarity = self._detect_polarity(smoothed)

        if max_abs_gradient < 3.0 and contrast < 4.0:
            return SnapResult(
                status="profile_too_flat",
                original_line=line,
                snapped_line=None,
                diameter_px=None,
                confidence=0.0,
                debug_payload={
                    "sample_count": int(len(positions)),
                    "profile_half_width": self.profile_half_width_px,
                    "polarity": polarity,
                    "peak_strength": 0.0,
                    "contrast": contrast,
                    "gradient_threshold": 3.0,
                    "used_fallback_pairing": False,
                    "angle_preserved": True,
                },
            )

        threshold = max(3.0, 0.25 * max_abs_gradient)
        edge_pair = self._find_edge_pair(gradient, threshold=threshold, polarity=polarity)
        if edge_pair is None:
            return SnapResult(
                status="edge_pair_not_found",
                original_line=line,
                snapped_line=None,
                diameter_px=None,
                confidence=0.0,
                debug_payload={
                    "sample_count": int(len(positions)),
                    "profile_half_width": self.profile_half_width_px,
                    "polarity": polarity,
                    "peak_strength": max_abs_gradient,
                    "contrast": contrast,
                    "gradient_threshold": threshold,
                    "used_fallback_pairing": False,
                    "angle_preserved": True,
                },
            )

        left_index, right_index, used_fallback = edge_pair
        left_position = self._subpixel_peak_position(positions, gradient, left_index)
        right_position = self._subpixel_peak_position(positions, gradient, right_index)
        if not math.isfinite(left_position) or not math.isfinite(right_position) or left_position >= right_position:
            return SnapResult(
                status="edge_pair_not_found",
                original_line=line,
                snapped_line=None,
                diameter_px=None,
                confidence=0.0,
                debug_payload={
                    "sample_count": int(len(positions)),
                    "profile_half_width": self.profile_half_width_px,
                    "polarity": polarity,
                    "left_peak_index": int(left_index),
                    "right_peak_index": int(right_index),
                    "peak_strength": max_abs_gradient,
                    "contrast": contrast,
                    "gradient_threshold": threshold,
                    "used_fallback_pairing": used_fallback,
                    "angle_preserved": True,
                },
            )

        start = self._line_point_at(line.start, axis, left_position, image_size)
        end = self._line_point_at(line.start, axis, right_position, image_size)
        snapped_line = Line(start=start, end=end)
        diameter_px = line_length(snapped_line)
        confidence = self._confidence_for_profile(
            gradient=gradient,
            left_index=left_index,
            right_index=right_index,
            contrast=contrast,
            left_position=left_position,
            right_position=right_position,
            total_length=float(positions[-1] - positions[0]) if len(positions) >= 2 else line_length(line),
        )
        status = "snapped" if confidence >= self.confidence_review_threshold else "manual_review"
        peak_strength = float((abs(float(gradient[left_index])) + abs(float(gradient[right_index]))) / 2.0)
        return SnapResult(
            status=status,
            original_line=line,
            snapped_line=snapped_line,
            diameter_px=diameter_px,
            confidence=confidence,
            debug_payload={
                "sample_count": int(len(positions)),
                "profile_half_width": self.profile_half_width_px,
                "polarity": polarity,
                "left_peak_index": int(left_index),
                "right_peak_index": int(right_index),
                "peak_strength": peak_strength,
                "contrast": contrast,
                "gradient_threshold": threshold,
                "used_fallback_pairing": used_fallback,
                "angle_preserved": True,
            },
        )

    def _to_grayscale_array(self, image: "QImage | RasterImage | np.ndarray[Any, Any]") -> np.ndarray[Any, np.float32]:
        if isinstance(image, RasterImage):
            if image.width <= 0 or image.height <= 0:
                return np.zeros((0, 0), dtype=np.float32)
            return np.asarray(image.pixels, dtype=np.float32).reshape((image.height, image.width))

        if isinstance(image, np.ndarray):
            array = np.asarray(image, dtype=np.float32)
            if array.ndim == 2:
                return array
            if array.ndim == 3 and array.shape[2] >= 3:
                return (0.299 * array[..., 0]) + (0.587 * array[..., 1]) + (0.114 * array[..., 2])
            raise ValueError("Unsupported ndarray shape for edge snapping.")

        if hasattr(image, "convertToFormat") and hasattr(image, "isNull"):
            if image.isNull():
                return np.zeros((0, 0), dtype=np.float32)
            from PySide6.QtGui import QImage

            grayscale = image.convertToFormat(QImage.Format.Format_Grayscale8)
            buffer = grayscale.constBits()
            array = np.frombuffer(buffer, dtype=np.uint8, count=grayscale.sizeInBytes())
            array = array.reshape((grayscale.height(), grayscale.bytesPerLine()))
            return array[:, : grayscale.width()].astype(np.float32, copy=True)

        raise TypeError("Unsupported image input for edge snapping.")

    def _extract_profile(
        self,
        grayscale: np.ndarray[Any, np.float32],
        line: Line,
    ) -> tuple[np.ndarray[Any, np.float32], np.ndarray[Any, np.float32], tuple[float, float], tuple[int, int]] | None:
        total_length = line_length(line)
        if total_length < 3.0:
            return None
        direction_axis = direction(line)
        normal_axis = (-direction_axis[1], direction_axis[0])
        sample_count = max(7, int(math.ceil(total_length / self.sample_step_px)) + 1)
        positions = np.linspace(0.0, total_length, num=sample_count, dtype=np.float32)
        base_x = line.start.x + (positions * direction_axis[0])
        base_y = line.start.y + (positions * direction_axis[1])
        offsets = np.arange(
            -float(self.profile_half_width_px),
            float(self.profile_half_width_px) + 1.0,
            1.0,
            dtype=np.float32,
        )
        sample_x = base_x[None, :] + (offsets[:, None] * normal_axis[0])
        sample_y = base_y[None, :] + (offsets[:, None] * normal_axis[1])
        profile = self._bilinear_sample(grayscale, sample_x, sample_y).mean(axis=0)
        if not np.isfinite(profile).all():
            return None
        return positions, profile.astype(np.float32, copy=False), direction_axis, (grayscale.shape[1], grayscale.shape[0])

    def _smooth_profile(self, profile: np.ndarray[Any, np.float32]) -> np.ndarray[Any, np.float32]:
        padded = np.pad(profile, (2, 2), mode="edge")
        return np.convolve(padded, self._gaussian_kernel, mode="valid").astype(np.float32, copy=False)

    def _require_numpy(self) -> None:
        if np is None:
            raise RuntimeError("numpy is required for the edge snap tool.")

    def _center_edge_contrast(self, profile: np.ndarray[Any, np.float32]) -> float:
        if profile.size == 0:
            return 0.0
        edge_count = max(1, int(profile.size * 0.15))
        center_count = max(1, int(profile.size * 0.30))
        center_start = max(0, (profile.size - center_count) // 2)
        center_end = min(profile.size, center_start + center_count)
        edge_values = np.concatenate([profile[:edge_count], profile[-edge_count:]])
        center_values = profile[center_start:center_end]
        if edge_values.size == 0 or center_values.size == 0:
            return 0.0
        return float(abs(float(center_values.mean()) - float(edge_values.mean())))

    def _detect_polarity(self, profile: np.ndarray[Any, np.float32]) -> str:
        if profile.size == 0:
            return "unknown"
        edge_count = max(1, int(profile.size * 0.15))
        center_count = max(1, int(profile.size * 0.30))
        center_start = max(0, (profile.size - center_count) // 2)
        center_end = min(profile.size, center_start + center_count)
        edge_values = np.concatenate([profile[:edge_count], profile[-edge_count:]])
        center_values = profile[center_start:center_end]
        if edge_values.size == 0 or center_values.size == 0:
            return "unknown"
        return "dark_on_light" if float(center_values.mean()) < float(edge_values.mean()) else "light_on_dark"

    def _find_edge_pair(
        self,
        gradient: np.ndarray[Any, np.float32],
        *,
        threshold: float,
        polarity: str,
    ) -> tuple[int, int, bool] | None:
        if gradient.size < 5:
            return None
        center = gradient.size // 2
        if polarity == "light_on_dark":
            left_sign = 1
            right_sign = -1
        else:
            left_sign = -1
            right_sign = 1

        left_peaks = self._signed_peaks(gradient[:center], sign=left_sign, threshold=threshold)
        right_peaks = [center + 1 + index for index in self._signed_peaks(gradient[center + 1 :], sign=right_sign, threshold=threshold)]
        if left_peaks and right_peaks:
            return left_peaks[-1], right_peaks[0], False

        left_fallback = self._strongest_signed_index(gradient[:center], sign=left_sign, threshold=threshold)
        right_fallback = self._strongest_signed_index(gradient[center + 1 :], sign=right_sign, threshold=threshold)
        if left_fallback is not None and right_fallback is not None:
            return left_fallback, center + 1 + right_fallback, True

        left_any = self._strongest_abs_index(gradient[:center], threshold=threshold)
        right_any = self._strongest_abs_index(gradient[center + 1 :], threshold=threshold)
        if left_any is None or right_any is None:
            return None
        left_index = left_any
        right_index = center + 1 + right_any
        if gradient[left_index] == 0 or gradient[right_index] == 0:
            return None
        if math.copysign(1.0, float(gradient[left_index])) == math.copysign(1.0, float(gradient[right_index])):
            return None
        return left_index, right_index, True

    def _signed_peaks(
        self,
        values: np.ndarray[Any, np.float32],
        *,
        sign: int,
        threshold: float,
    ) -> list[int]:
        peaks: list[int] = []
        if values.size < 3:
            return peaks
        for index in range(1, values.size - 1):
            current = float(values[index])
            previous = float(values[index - 1])
            nxt = float(values[index + 1])
            if sign > 0:
                if current >= threshold and current >= previous and current >= nxt:
                    peaks.append(index)
            else:
                if current <= -threshold and current <= previous and current <= nxt:
                    peaks.append(index)
        return peaks

    def _strongest_signed_index(
        self,
        values: np.ndarray[Any, np.float32],
        *,
        sign: int,
        threshold: float,
    ) -> int | None:
        if values.size == 0:
            return None
        mask = values >= threshold if sign > 0 else values <= -threshold
        candidates = np.nonzero(mask)[0]
        if candidates.size == 0:
            return None
        strengths = np.abs(values[candidates])
        return int(candidates[int(np.argmax(strengths))])

    def _strongest_abs_index(self, values: np.ndarray[Any, np.float32], *, threshold: float) -> int | None:
        if values.size == 0:
            return None
        strengths = np.abs(values)
        index = int(np.argmax(strengths))
        return index if float(strengths[index]) >= threshold else None

    def _subpixel_peak_position(
        self,
        positions: np.ndarray[Any, np.float32],
        gradient: np.ndarray[Any, np.float32],
        index: int,
    ) -> float:
        if index <= 0 or index >= len(gradient) - 1:
            return float(positions[index])
        y0 = float(gradient[index - 1])
        y1 = float(gradient[index])
        y2 = float(gradient[index + 1])
        denominator = y0 - (2.0 * y1) + y2
        if abs(denominator) < 1e-9:
            return float(positions[index])
        offset = 0.5 * (y0 - y2) / denominator
        offset = max(-1.0, min(1.0, offset))
        step = float(positions[index] - positions[index - 1])
        return float(positions[index] + (offset * step))

    def _confidence_for_profile(
        self,
        *,
        gradient: np.ndarray[Any, np.float32],
        left_index: int,
        right_index: int,
        contrast: float,
        left_position: float,
        right_position: float,
        total_length: float,
    ) -> float:
        peak_strength = (abs(float(gradient[left_index])) + abs(float(gradient[right_index]))) / 2.0
        peak_score = min(1.0, peak_strength / 20.0)
        contrast_score = min(1.0, contrast / 24.0)
        midpoint_position = (left_position + right_position) / 2.0
        symmetry = max(0.0, 1.0 - (abs(midpoint_position - (total_length / 2.0)) / max(total_length / 2.0, 1.0)))
        return max(0.01, min(0.99, (peak_score * 0.45) + (symmetry * 0.25) + (contrast_score * 0.30)))

    def _line_point_at(
        self,
        start: Point,
        axis: tuple[float, float],
        distance_along_line: float,
        image_size: tuple[int, int],
    ) -> Point:
        width, height = image_size
        return Point(
            x=clamp(start.x + (axis[0] * distance_along_line), 0.0, max(0.0, width - 1.0)),
            y=clamp(start.y + (axis[1] * distance_along_line), 0.0, max(0.0, height - 1.0)),
        )

    def _bilinear_sample(
        self,
        image: np.ndarray[Any, np.float32],
        xs: np.ndarray[Any, np.float32],
        ys: np.ndarray[Any, np.float32],
    ) -> np.ndarray[Any, np.float32]:
        height, width = image.shape
        xs = np.clip(xs, 0.0, max(0.0, width - 1.0))
        ys = np.clip(ys, 0.0, max(0.0, height - 1.0))
        x0 = np.floor(xs).astype(np.int32)
        y0 = np.floor(ys).astype(np.int32)
        x1 = x0 + 1
        y1 = y0 + 1

        x0 = np.clip(x0, 0, width - 1)
        x1 = np.clip(x1, 0, width - 1)
        y0 = np.clip(y0, 0, height - 1)
        y1 = np.clip(y1, 0, height - 1)

        x_weight = xs - x0
        y_weight = ys - y0

        top_left = image[y0, x0]
        top_right = image[y0, x1]
        bottom_left = image[y1, x0]
        bottom_right = image[y1, x1]

        top = (top_left * (1.0 - x_weight)) + (top_right * x_weight)
        bottom = (bottom_left * (1.0 - x_weight)) + (bottom_right * x_weight)
        return (top * (1.0 - y_weight)) + (bottom * y_weight)
