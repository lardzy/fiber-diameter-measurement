"""Pure, cancellable 2-D metrology fitting on original image samples.

Coordinates passed to this module are image coordinates, never widget
coordinates.  ``scale`` maps one image pixel to one physical unit along X/Y.
The module has no Qt dependency and does not modify its input image.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math
from typing import Any, Callable, Iterable, Sequence

import cv2
import numpy as np


PointTuple = tuple[float, float]
CancelCheck = Callable[[], bool] | None
ALGORITHM_VERSION = "metrology-caliper/1"


@dataclass(frozen=True, slots=True)
class CaliperOptions:
    method: str = "gradient"
    polarity: str = "either"
    reverse_search: bool = False
    sample_step_px: float = 0.5
    search_half_width_px: float = 10.0
    averaging_half_width_px: float = 2.0
    sample_count: int = 32
    threshold_percent: float = 50.0
    min_contrast: float = 4.0 / 255.0
    min_gradient: float = 3.0 / 255.0
    min_valid_fraction: float = 0.5
    robust: bool = True
    max_residual_px: float = 2.0
    min_circle_coverage_deg: float = 90.0

    def __post_init__(self) -> None:
        if self.method not in {"gradient", "threshold"}:
            raise ValueError("边缘模式必须为 gradient 或 threshold")
        if self.polarity not in {"either", "rising", "falling"}:
            raise ValueError("边缘极性必须为 either、rising 或 falling")
        if not isinstance(self.reverse_search, bool) or not isinstance(self.robust, bool):
            raise ValueError("反向搜索和稳健拟合参数必须为布尔值")
        limits = {
            "sample_step_px": (0.125, 4.0),
            "search_half_width_px": (1.0, 1024.0),
            "averaging_half_width_px": (0.0, 64.0),
            "threshold_percent": (0.1, 99.9),
            "min_contrast": (0.0, 1.0),
            "min_gradient": (0.0, 2.0),
            "min_valid_fraction": (0.0, 1.0),
            "max_residual_px": (0.0, 1024.0),
            "min_circle_coverage_deg": (0.0, 360.0),
        }
        for name, (low, high) in limits.items():
            value = float(getattr(self, name))
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} 必须在 {low}～{high} 范围内")
            object.__setattr__(self, name, value)
        if isinstance(self.sample_count, bool) or int(self.sample_count) != self.sample_count:
            raise ValueError("采样截面数量必须为整数")
        if not 3 <= self.sample_count <= 1024:
            raise ValueError("采样截面数量必须在 3～1024 范围内")
        object.__setattr__(self, "sample_count", int(self.sample_count))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CaliperOptions":
        unknown = set(payload) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"未知卡尺参数: {', '.join(sorted(unknown))}")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class EdgeCandidate:
    offset_px: float
    point_px: PointTuple
    polarity: str
    strength: float


@dataclass(frozen=True, slots=True)
class CaliperProfile:
    index: int
    center_px: PointTuple
    search_polygon_px: tuple[PointTuple, ...]
    positions_px: tuple[float, ...] = ()
    intensities: tuple[float, ...] = ()
    candidates: tuple[EdgeCandidate, ...] = ()
    selected_indices: tuple[int, ...] = ()
    status: str = "missing_edge"
    reason: str = ""

    @property
    def selected_points_px(self) -> tuple[PointTuple, ...]:
        return tuple(self.candidates[i].point_px for i in self.selected_indices)


@dataclass(frozen=True, slots=True)
class CaliperSection:
    index: int
    first_px: PointTuple
    second_px: PointTuple
    position: float
    width: float
    accepted: bool = True
    reason: str = ""


@dataclass(frozen=True, slots=True)
class FitResult:
    kind: str
    status: str
    reason: str = ""
    geometry: dict[str, Any] = field(default_factory=dict)
    geometry_px: dict[str, Any] = field(default_factory=dict)
    points_px: tuple[PointTuple, ...] = ()
    accepted_indices: tuple[int, ...] = ()
    rejected_indices: tuple[int, ...] = ()
    residuals: tuple[float, ...] = ()
    rms: float | None = None
    max_residual: float | None = None
    coverage_deg: float | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == "ready"

    @property
    def accepted_points_px(self) -> tuple[PointTuple, ...]:
        return tuple(self.points_px[i] for i in self.accepted_indices)

    @property
    def rejected_points_px(self) -> tuple[PointTuple, ...]:
        return tuple(self.points_px[i] for i in self.rejected_indices)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CaliperResult:
    kind: str
    status: str
    reason: str = ""
    fit: FitResult | None = None
    profiles: tuple[CaliperProfile, ...] = ()
    sections: tuple[CaliperSection, ...] = ()
    statistics: dict[str, float | int] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)
    scale: PointTuple = (1.0, 1.0)
    algorithm_version: str = ALGORITHM_VERSION

    @property
    def success(self) -> bool:
        return self.status == "ready"

    @property
    def valid_fraction(self) -> float:
        if not self.profiles:
            return 0.0
        return sum(p.status == "ready" for p in self.profiles) / len(self.profiles)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _Cancelled(Exception):
    pass


def _check_cancel(cancel: CancelCheck) -> None:
    if cancel is not None and cancel():
        raise _Cancelled


def _point(value: Any) -> PointTuple:
    if hasattr(value, "x") and hasattr(value, "y"):
        result = float(value.x), float(value.y)
    else:
        result = float(value[0]), float(value[1])
    if not all(math.isfinite(v) for v in result):
        raise ValueError("坐标必须为有限数")
    return result


def _scale(value: Sequence[float]) -> np.ndarray:
    values = np.asarray(_point(value), dtype=np.float64)
    if np.any(values <= 0):
        raise ValueError("X/Y 标定比例必须是正有限数")
    return values


def _points(points: Iterable[Any], minimum: int) -> np.ndarray:
    result = np.asarray([_point(p) for p in points], dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 2 or len(result) < minimum:
        raise ValueError(f"至少需要 {minimum} 个有效点")
    if len(result) > 100_000:
        raise ValueError("单次拟合最多支持 100000 个点")
    if len(np.unique(result, axis=0)) < minimum:
        raise ValueError(f"至少需要 {minimum} 个不同的点")
    return result


def _tuples(points: np.ndarray) -> tuple[PointTuple, ...]:
    return tuple((float(x), float(y)) for x, y in points)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= np.finfo(float).eps:
        raise ValueError("方向长度为零")
    return vector / norm


def _inlier_mask(residuals: np.ndarray, extent: float, minimum: int) -> np.ndarray:
    magnitude = np.abs(residuals)
    median = float(np.median(magnitude))
    mad = float(np.median(np.abs(magnitude - median)))
    cutoff = max(median + 3.5 * 1.4826 * mad, extent * 1e-8, 1e-12)
    mask = magnitude <= cutoff
    if int(mask.sum()) < minimum:
        raise ValueError("有效拟合点不足")
    return mask


def _line_solution(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    _u, singular, vh = np.linalg.svd(points - center, full_matrices=False)
    if singular[0] <= max(1e-14, np.max(np.abs(points - center)) * 1e-12):
        raise ValueError("点重合，无法拟合直线")
    direction = vh[0]
    if np.dot(direction, points[-1] - points[0]) < 0:
        direction = -direction
    return center, direction


def _line_residual(points: np.ndarray, center: np.ndarray, axis: np.ndarray) -> np.ndarray:
    return (points - center) @ np.array([-axis[1], axis[0]])


def _robust_line_seed(points: np.ndarray, cancel: CancelCheck) -> tuple[np.ndarray, np.ndarray]:
    best = _line_solution(points)
    best_score = float(np.median(np.abs(_line_residual(points, *best))))
    rng = np.random.default_rng(0)
    for _ in range(min(192, len(points) * 8)):
        _check_cancel(cancel)
        indices = rng.choice(len(points), 2, replace=False)
        delta = points[indices[1]] - points[indices[0]]
        if np.linalg.norm(delta) <= 1e-12:
            continue
        candidate = points[indices[0]], _unit(delta)
        score = float(np.median(np.abs(_line_residual(points, *candidate))))
        if score < best_score:
            best, best_score = candidate, score
    return best


def fit_line_points(
    points: Iterable[Any], *, scale: Sequence[float] = (1.0, 1.0),
    robust: bool = True, cancel: CancelCheck = None,
) -> FitResult:
    """Fit an orthogonal-distance line; returned residuals use physical units."""
    try:
        _check_cancel(cancel)
        source = _points(points, 2)
        factors = _scale(scale)
        origin = source.mean(axis=0) * factors
        physical = source * factors - origin
        extent = max(float(np.ptp(physical, axis=0).max()), 1e-12)
        mask = np.ones(len(source), dtype=bool)
        if robust and len(source) >= 5:
            center, axis = _robust_line_seed(physical, cancel)
            for _ in range(5):
                _check_cancel(cancel)
                updated = _inlier_mask(_line_residual(physical, center, axis), extent, 2)
                center, axis = _line_solution(physical[updated])
                if np.array_equal(updated, mask):
                    mask = updated
                    break
                mask = updated
        else:
            center, axis = _line_solution(physical)
        _check_cancel(cancel)
        accepted_points = physical[mask]
        if np.dot(axis, accepted_points[-1] - accepted_points[0]) < 0:
            axis = -axis
        residuals = _line_residual(physical, center, axis)
        projection = (physical[mask] - center) @ axis
        ends = np.vstack((center + projection.min() * axis, center + projection.max() * axis)) + origin
        global_center = center + origin
        accepted = np.flatnonzero(mask)
        return FitResult(
            "line", "ready",
            geometry={"point": tuple(global_center), "direction": tuple(axis), "endpoints": _tuples(ends)},
            geometry_px={"point": tuple(global_center / factors), "direction": tuple(_unit(axis / factors)), "endpoints": _tuples(ends / factors)},
            points_px=_tuples(source), accepted_indices=tuple(map(int, accepted)),
            rejected_indices=tuple(map(int, np.flatnonzero(~mask))),
            residuals=tuple(map(float, residuals)),
            rms=float(np.sqrt(np.mean(residuals[mask] ** 2))),
            max_residual=float(np.max(np.abs(residuals[mask]))),
            diagnostics={"algorithm": "orthogonal-svd", "robust": bool(robust), "scale": tuple(map(float, factors)), "version": ALGORITHM_VERSION},
        )
    except _Cancelled:
        return FitResult("line", "cancelled", "拟合已取消")
    except (ValueError, TypeError, IndexError, np.linalg.LinAlgError, OverflowError) as exc:
        return FitResult("line", "invalid_input", str(exc))


def _algebraic_circle(points: np.ndarray) -> tuple[np.ndarray, float]:
    mean = points.mean(axis=0)
    normalized = points - mean
    extent = max(float(np.max(np.linalg.norm(normalized, axis=1))), 1e-15)
    normalized /= extent
    matrix = np.column_stack((2 * normalized, np.ones(len(points))))
    rhs = np.sum(normalized * normalized, axis=1)
    solution, _residuals, rank, singular = np.linalg.lstsq(matrix, rhs, rcond=None)
    if rank < 3 or singular[-1] < singular[0] * 1e-10:
        raise ValueError("点共线或圆拟合条件不足")
    radius_sq = float(solution[2] + np.dot(solution[:2], solution[:2]))
    if radius_sq <= 0 or not math.isfinite(radius_sq):
        raise ValueError("圆半径无效")
    return solution[:2] * extent + mean, math.sqrt(radius_sq) * extent


def _circle_residual(points: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    return np.linalg.norm(points - center, axis=1) - radius


def _robust_circle_seed(points: np.ndarray, cancel: CancelCheck) -> tuple[np.ndarray, float]:
    best = _algebraic_circle(points)
    score = float(np.median(np.abs(_circle_residual(points, *best))))
    rng = np.random.default_rng(0)
    for _ in range(min(256, len(points) * 8)):
        _check_cancel(cancel)
        indices = rng.choice(len(points), 3, replace=False)
        try:
            candidate = _algebraic_circle(points[indices])
        except ValueError:
            continue
        candidate_score = float(np.median(np.abs(_circle_residual(points, *candidate))))
        if candidate_score < score:
            best, score = candidate, candidate_score
    return best


def _refine_circle(
    points: np.ndarray, center: np.ndarray, radius: float, cancel: CancelCheck,
) -> tuple[np.ndarray, float]:
    center = center.copy()
    for _ in range(40):
        _check_cancel(cancel)
        delta = center - points
        distances = np.linalg.norm(delta, axis=1)
        if np.any(distances <= max(radius * 1e-12, 1e-15)):
            raise ValueError("采样点与拟合圆心重合")
        residuals = distances - radius
        jacobian = np.column_stack((delta / distances[:, None], -np.ones(len(points))))
        update, _sse, rank, _singular = np.linalg.lstsq(jacobian, -residuals, rcond=None)
        if rank < 3:
            raise ValueError("圆拟合退化")
        # Backtracking prevents a poorly initialized circle taking an unbounded step.
        objective = float(np.dot(residuals, residuals))
        factor = 1.0
        for _ in range(12):
            new_center = center + factor * update[:2]
            new_radius = radius + factor * update[2]
            if new_radius > 0:
                candidate = _circle_residual(points, new_center, new_radius)
                if float(np.dot(candidate, candidate)) <= objective + 1e-24:
                    break
            factor *= 0.5
        else:
            break
        center, radius = new_center, float(new_radius)
        if np.linalg.norm(factor * update) <= max(radius * 1e-11, 1e-13):
            break
    if radius <= 0 or not math.isfinite(radius) or not np.isfinite(center).all():
        raise ValueError("圆拟合未得到有限解")
    return center, radius


def _coverage(points: np.ndarray, center: np.ndarray) -> float:
    angles = np.sort(np.mod(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]), 2 * math.pi))
    gaps = np.diff(np.r_[angles, angles[0] + 2 * math.pi])
    return float(math.degrees(2 * math.pi - np.max(gaps)))


def fit_circle_points(
    points: Iterable[Any], *, scale: Sequence[float] = (1.0, 1.0),
    robust: bool = True, cancel: CancelCheck = None,
) -> FitResult:
    """Fit in the calibrated plane; an anisotropic image gets elliptical radii."""
    try:
        _check_cancel(cancel)
        source = _points(points, 3)
        factors = _scale(scale)
        origin = source.mean(axis=0) * factors
        physical = source * factors - origin
        _u, singular, _vh = np.linalg.svd(physical - physical.mean(axis=0), full_matrices=False)
        if singular[-1] <= singular[0] * 1e-8:
            raise ValueError("点共线或近共线，无法定义圆")
        extent = max(float(np.ptp(physical, axis=0).max()), 1e-12)
        center, radius = _robust_circle_seed(physical, cancel) if robust and len(source) >= 6 else _algebraic_circle(physical)
        mask = np.ones(len(source), dtype=bool)
        if robust and len(source) >= 6:
            for _ in range(5):
                _check_cancel(cancel)
                updated = _inlier_mask(_circle_residual(physical, center, radius), extent, 3)
                center, radius = _refine_circle(physical[updated], center, radius, cancel)
                if np.array_equal(updated, mask):
                    mask = updated
                    break
                mask = updated
        else:
            center, radius = _refine_circle(physical, center, radius, cancel)
        residuals = _circle_residual(physical, center, radius)
        global_center = center + origin
        return FitResult(
            "circle", "ready", geometry={"center": tuple(global_center), "radius": radius},
            geometry_px={"center": tuple(global_center / factors), "radii": tuple(radius / factors)},
            points_px=_tuples(source), accepted_indices=tuple(map(int, np.flatnonzero(mask))),
            rejected_indices=tuple(map(int, np.flatnonzero(~mask))),
            residuals=tuple(map(float, residuals)), rms=float(np.sqrt(np.mean(residuals[mask] ** 2))),
            max_residual=float(np.max(np.abs(residuals[mask]))), coverage_deg=_coverage(physical[mask], center),
            diagnostics={"algorithm": "algebraic-initial-geometric-circle", "robust": bool(robust), "scale": tuple(map(float, factors)), "version": ALGORITHM_VERSION},
        )
    except _Cancelled:
        return FitResult("circle", "cancelled", "拟合已取消")
    except (ValueError, TypeError, IndexError, np.linalg.LinAlgError, OverflowError) as exc:
        return FitResult("circle", "invalid_input", str(exc))


def _grayscale(image: np.ndarray, channel_order: str) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim not in {2, 3} or array.shape[0] < 2 or array.shape[1] < 2:
        raise ValueError("图像必须为至少 2×2 的灰度或彩色数组")
    if array.dtype not in {np.dtype("uint8"), np.dtype("uint16"), np.dtype("float32"), np.dtype("float64")}:
        raise ValueError("支持 uint8、uint16 和 [0,1] 浮点原图")
    data = array.astype(np.float32, copy=True)
    if array.dtype.kind == "u":
        data /= float(np.iinfo(array.dtype).max)
    elif not np.isfinite(data).all() or float(data.min()) < 0 or float(data.max()) > 1:
        raise ValueError("浮点图像强度必须在 [0,1] 内")
    order = str(channel_order).upper()
    if data.ndim == 2:
        if order not in {"RGB", "BGR", "GRAY"}:
            raise ValueError("灰度数组与通道顺序不匹配")
        return data
    if order not in {"RGB", "BGR", "RGBA", "BGRA"} or data.shape[2] != len(order):
        raise ValueError("必须显式提供与彩色数组匹配的 RGB/BGR/RGBA/BGRA 顺序")
    indices = [order.index(c) for c in "RGB"]
    gray = data[..., indices] @ np.asarray([0.299, 0.587, 0.114], np.float32)
    # Fully/partly transparent pixels contain no unambiguous original RGB sample.
    if "A" in order:
        gray[data[..., order.index("A")] < 1.0] = np.nan
    return np.ascontiguousarray(gray)


def _smooth_profile(profile: np.ndarray) -> np.ndarray:
    kernel = np.array([1, 4, 6, 4, 1], dtype=np.float64) / 16
    return np.convolve(np.pad(profile, (2, 2), mode="edge"), kernel, mode="valid")


def _edge_candidates(
    positions: np.ndarray, profile: np.ndarray, center: np.ndarray, normal: np.ndarray,
    options: CaliperOptions,
) -> tuple[EdgeCandidate, ...]:
    smoothed = _smooth_profile(profile)
    if float(np.ptp(smoothed)) < max(options.min_contrast, 1e-12):
        return ()
    edges: list[EdgeCandidate] = []
    if options.method == "threshold":
        threshold = float(smoothed.min() + np.ptp(smoothed) * options.threshold_percent / 100.0)
        for i, (left, right) in enumerate(zip(smoothed[:-1], smoothed[1:])):
            slope = float(right - left)
            if slope == 0 or not (min(left, right) <= threshold <= max(left, right)):
                continue
            fraction = float((threshold - left) / slope)
            position = float(positions[i] + fraction * (positions[i + 1] - positions[i]))
            strength = abs(slope / (positions[i + 1] - positions[i]))
            if strength < options.min_gradient:
                continue
            if edges and abs(position - edges[-1].offset_px) < options.sample_step_px * 0.1:
                continue
            edges.append(EdgeCandidate(position, tuple(center + position * normal), "rising" if slope > 0 else "falling", strength))
    else:
        gradient = np.gradient(smoothed, positions)
        magnitude = np.abs(gradient)
        i = 1
        while i < len(gradient) - 1:
            if magnitude[i] < options.min_gradient or magnitude[i] < magnitude[i - 1] or magnitude[i] < magnitude[i + 1]:
                i += 1
                continue
            end = i
            while end + 1 < len(gradient) - 1 and abs(magnitude[end + 1] - magnitude[i]) <= 1e-12:
                end += 1
            if end > i:
                index = (i + end) / 2.0
                position = float(np.interp(index, np.arange(len(positions)), positions))
                sign = gradient[i]
                strength = float(magnitude[i])
            else:
                denominator = magnitude[i - 1] - 2 * magnitude[i] + magnitude[i + 1]
                shift = 0.0 if abs(denominator) <= 1e-15 else float(0.5 * (magnitude[i - 1] - magnitude[i + 1]) / denominator)
                shift = max(-0.5, min(0.5, shift))
                position = float(positions[i] + shift * (positions[i + 1] - positions[i]))
                sign = gradient[i]
                strength = float(magnitude[i])
            edges.append(EdgeCandidate(position, tuple(center + position * normal), "rising" if sign > 0 else "falling", strength))
            i = end + 1
    return tuple(edges)


def _select_edges(candidates: tuple[EdgeCandidate, ...], options: CaliperOptions, double_edge: bool) -> tuple[int, ...]:
    valid = [i for i, candidate in enumerate(candidates) if options.polarity == "either" or candidate.polarity == options.polarity]
    if not double_edge:
        return (min(valid, key=lambda i: (abs(candidates[i].offset_px), -candidates[i].strength)),) if valid else ()
    # A width is an edge pair straddling the explicitly chosen center line.
    # The selected polarity applies to the first edge along the search direction.
    pairs = [
        (i, j) for i in valid for j in range(len(candidates))
        if candidates[i].offset_px < 0 < candidates[j].offset_px
        and candidates[i].polarity != candidates[j].polarity
    ]
    if not pairs:
        return ()
    return min(pairs, key=lambda pair: abs(candidates[pair[0]].offset_px) + abs(candidates[pair[1]].offset_px))


def _profile(
    gray: np.ndarray, index: int, center: np.ndarray, normal: np.ndarray,
    tangent: np.ndarray, origin: np.ndarray, options: CaliperOptions, double_edge: bool,
) -> CaliperProfile:
    normal = normal * (-1 if options.reverse_search else 1)
    half_width = options.search_half_width_px
    half_strip = options.averaging_half_width_px
    polygon = _tuples(np.asarray([
        center - half_width * normal - half_strip * tangent,
        center + half_width * normal - half_strip * tangent,
        center + half_width * normal + half_strip * tangent,
        center - half_width * normal + half_strip * tangent,
    ]))
    count = max(5, int(math.ceil(half_width * 2 / options.sample_step_px)) + 1)
    positions = np.linspace(-half_width, half_width, count)
    strip = np.linspace(-half_strip, half_strip, max(1, int(math.ceil(half_strip * 2)) + 1))
    locations = center[None, None, :] + positions[None, :, None] * normal + strip[:, None, None] * tangent - origin
    if (locations[..., 0].min() < 0 or locations[..., 1].min() < 0
            or locations[..., 0].max() > gray.shape[1] - 1 or locations[..., 1].max() > gray.shape[0] - 1):
        return CaliperProfile(index, tuple(center), polygon, status="out_of_bounds", reason="完整搜索截面超出原始图像范围")
    sampled = cv2.remap(gray, locations[..., 0].astype(np.float32), locations[..., 1].astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if not np.isfinite(sampled).all():
        return CaliperProfile(index, tuple(center), polygon, status="invalid_pixels", reason="搜索区包含透明或无效像素")
    intensities = sampled.mean(axis=0, dtype=np.float64)
    candidates = _edge_candidates(positions, intensities, center, normal, options)
    selected = _select_edges(candidates, options, double_edge)
    reason = "" if selected else ("未找到跨越基准的双边" if double_edge else "未找到符合方向和对比度的边缘")
    return CaliperProfile(index, tuple(center), polygon, tuple(map(float, positions)), tuple(map(float, intensities)), candidates, selected, "ready" if selected else "missing_edge", reason)


def _finish_caliper(
    kind: str, profiles: list[CaliperProfile], options: CaliperOptions,
    factors: np.ndarray, cancel: CancelCheck, sections: list[CaliperSection] | None = None,
) -> CaliperResult:
    sections = [] if sections is None else sections
    good_indices = [i for i, p in enumerate(profiles) if p.status == "ready"]
    minimum = 3 if kind == "circle" else 2
    initial_statistics = {
        "sample_count": len(profiles), "valid_count": len(good_indices),
        "valid_fraction": len(good_indices) / len(profiles) if profiles else 0.0,
    }
    common = {"profiles": tuple(profiles), "options": options.to_dict(), "scale": tuple(map(float, factors)), "sections": tuple(sections), "statistics": initial_statistics}
    if len(good_indices) < minimum:
        return CaliperResult(kind, "insufficient_edges", f"有效截面不足，需要至少 {minimum} 个", **common)
    if len(good_indices) / len(profiles) < options.min_valid_fraction:
        return CaliperResult(kind, "low_valid_fraction", "有效边缘截面比例不足", **common)
    if sections:
        selected_points = [tuple((np.asarray(s.first_px) + s.second_px) / 2) for s in sections]
    else:
        selected_points = [profiles[i].selected_points_px[0] for i in good_indices]
    fit = (fit_circle_points if kind == "circle" else fit_line_points)(selected_points, scale=factors, robust=options.robust, cancel=cancel)
    if not fit.success:
        return CaliperResult(kind, fit.status, fit.reason, fit=fit, **common)
    rejected = set(fit.rejected_indices)
    if sections and options.robust and len(sections) >= 5:
        widths = np.asarray([s.width for s in sections])
        width_mask = _inlier_mask(widths - np.median(widths), max(float(np.max(widths)), 1e-12), 2)
        rejected.update(map(int, np.flatnonzero(~width_mask)))
    for local_index in rejected:
        profile_index = good_indices[local_index]
        profiles[profile_index] = replace(profiles[profile_index], status="rejected", reason="稳健拟合剔除的异常截面")
    if sections:
        sections = [replace(s, accepted=i not in rejected, reason="异常截面" if i in rejected else "") for i, s in enumerate(sections)]
    statistics: dict[str, float | int] = {
        "sample_count": len(profiles), "valid_count": len(good_indices) - len(rejected),
        "valid_fraction": (len(good_indices) - len(rejected)) / len(profiles),
    }
    if sections:
        widths = np.asarray([s.width for s in sections if s.accepted])
        if len(widths):
            statistics.update({"minimum": float(widths.min()), "maximum": float(widths.max()), "mean": float(widths.mean()), "stddev": float(widths.std(ddof=1)) if len(widths) > 1 else 0.0, "count": len(widths)})
    status, reason = "ready", ""
    if statistics["valid_fraction"] < options.min_valid_fraction:
        status, reason = "low_valid_fraction", "剔除异常点后的有效截面比例不足"
    # Conservative conversion: a physical residual must be small even on the finer axis.
    elif fit.max_residual is not None and fit.max_residual / float(factors.min()) > options.max_residual_px:
        status, reason = "poor_fit", "拟合残差超过设置的像素阈值，请复核采样点"
    elif kind == "circle" and float(fit.coverage_deg or 0) < options.min_circle_coverage_deg:
        status, reason = "insufficient_coverage", "有效圆周覆盖角不足，请扩大搜索范围"
    return CaliperResult(kind, status, reason, fit=fit, profiles=tuple(profiles), sections=tuple(sections), statistics=statistics, options=options.to_dict(), scale=tuple(map(float, factors)))


def run_line_caliper(
    image: np.ndarray, start_px: Any, end_px: Any, *, options: CaliperOptions | None = None,
    scale: Sequence[float] = (1.0, 1.0), channel_order: str = "RGB",
    origin_px: Any = (0.0, 0.0), double_edge: bool = False, cancel: CancelCheck = None,
) -> CaliperResult:
    """Search perpendicular to a baseline. Double-edge mode measures thickness.

    For cropped images, all points remain global; ``origin_px`` is the crop's
    global top-left. Search widths/averaging widths are expressed in pixels.
    """
    profiles: list[CaliperProfile] = []
    sections: list[CaliperSection] = []
    kind = "double_edge" if double_edge else "line"
    options = CaliperOptions() if options is None else options
    factors = np.ones(2)
    try:
        _check_cancel(cancel)
        factors = _scale(scale)
        gray = _grayscale(image, channel_order)
        start, end, origin = np.asarray(_point(start_px)), np.asarray(_point(end_px)), np.asarray(_point(origin_px))
        tangent = _unit(end - start)
        physical_axis = _unit((end - start) * factors)
        normal = _unit(np.array([-physical_axis[1], physical_axis[0]]) / factors)
        for index, fraction in enumerate(np.linspace(0, 1, options.sample_count)):
            _check_cancel(cancel)
            center = start + fraction * (end - start)
            profile = _profile(gray, index, center, normal, tangent, origin, options, double_edge)
            profiles.append(profile)
            if double_edge and profile.status == "ready":
                first, second = profile.selected_points_px
                width = float(np.linalg.norm((np.asarray(second) - first) * factors))
                sections.append(CaliperSection(index, first, second, float(np.linalg.norm((center - start) * factors)), width))
        return _finish_caliper(kind, profiles, options, factors, cancel, sections)
    except _Cancelled:
        return CaliperResult(kind, "cancelled", "卡尺测量已取消", profiles=tuple(profiles), sections=tuple(sections), options=options.to_dict(), scale=tuple(map(float, factors)))
    except (ValueError, TypeError, IndexError, OverflowError, np.linalg.LinAlgError, cv2.error) as exc:
        return CaliperResult(kind, "invalid_input", str(exc), profiles=tuple(profiles), sections=tuple(sections), options=options.to_dict(), scale=tuple(map(float, factors)))


def run_circle_caliper(
    image: np.ndarray, center_px: Any, radius_px: float, *, radius_y_px: float | None = None,
    start_angle_deg: float = 0.0, sweep_angle_deg: float = 360.0,
    options: CaliperOptions | None = None, scale: Sequence[float] = (1.0, 1.0),
    channel_order: str = "RGB", origin_px: Any = (0.0, 0.0), cancel: CancelCheck = None,
) -> CaliperResult:
    """Radial calipers; use separate X/Y image radii for anisotropic calibration."""
    profiles: list[CaliperProfile] = []
    options = CaliperOptions() if options is None else options
    factors = np.ones(2)
    try:
        _check_cancel(cancel)
        factors = _scale(scale)
        gray = _grayscale(image, channel_order)
        center, origin = np.asarray(_point(center_px)), np.asarray(_point(origin_px))
        rx = float(radius_px)
        ry = rx if radius_y_px is None else float(radius_y_px)
        if not all(math.isfinite(v) for v in (rx, ry, start_angle_deg, sweep_angle_deg)) or min(rx, ry) <= 0 or not 0 < sweep_angle_deg <= 360:
            raise ValueError("半径必须为正有限数，扫描角必须在 (0,360] 内")
        angles = np.radians(np.linspace(start_angle_deg, start_angle_deg + sweep_angle_deg, options.sample_count, endpoint=sweep_angle_deg < 360))
        for index, angle in enumerate(angles):
            _check_cancel(cancel)
            radial = np.array([rx * math.cos(angle), ry * math.sin(angle)])
            physical_radial = radial * factors
            normal = _unit(radial)
            tangent = _unit(np.array([-physical_radial[1], physical_radial[0]]) / factors)
            profiles.append(_profile(gray, index, center + radial, normal, tangent, origin, options, False))
        return _finish_caliper("circle", profiles, options, factors, cancel)
    except _Cancelled:
        return CaliperResult("circle", "cancelled", "卡尺测量已取消", profiles=tuple(profiles), options=options.to_dict(), scale=tuple(map(float, factors)))
    except (ValueError, TypeError, IndexError, OverflowError, np.linalg.LinAlgError, cv2.error) as exc:
        return CaliperResult("circle", "invalid_input", str(exc), profiles=tuple(profiles), options=options.to_dict(), scale=tuple(map(float, factors)))
