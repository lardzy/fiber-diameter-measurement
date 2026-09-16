from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import ceil, hypot, isfinite
from time import perf_counter
from typing import Callable

import cv2

from fdm.geometry import Line, Point, distance
from fdm.services.prompt_segmentation import (
    fill_magic_draft_internal_holes,
    magic_mask_to_geometry,
    normalize_magic_draft_mask,
)

DEFAULT_FIBER_QUICK_GEOMETRY_TIMEOUT_MS = 3000.0
FIBER_QUICK_SKELETON_BACKEND = "skimage_zhang"
FIBER_QUICK_GEOMETRY_REVISION = 3


class FiberQuickGeometryError(RuntimeError):
    """An actionable message plus machine-readable reasons for diagnostics."""

    def __init__(self, code: str, message: str, debug_payload: dict[str, object]) -> None:
        super().__init__(message)
        self.code = code
        self.debug_payload = {**debug_payload, "failure_code": code}


@lru_cache(maxsize=1)
def prepare_fiber_quick_geometry_backend():
    """Load the same compiled implementation in development and frozen builds.

    Kept lazy so the Qt worker can warm it before segmentation completes.
    A missing/broken binary dependency must not silently change the algorithm.
    """
    try:
        from skimage.morphology import skeletonize
    except (ImportError, OSError) as exc:
        raise RuntimeError("快速测径骨架组件不可用，请修复安装（scikit-image）。") from exc
    return skeletonize


@dataclass(slots=True)
class FiberQuickDiameterGeometryResult:
    line_px: Line | None
    confidence: float
    status: str
    preview_polygon_px: list[Point]
    preview_area_rings_px: list[list[Point]]
    debug_payload: dict[str, object]


class FiberQuickDiameterGeometryService:
    def measure_from_mask(
        self,
        mask,
        *,
        positive_points: list[Point] | None = None,
        negative_points: list[Point] | None = None,
        preview_polygon_points: list[Point] | None = None,
        preview_area_rings_points: list[list[Point]] | None = None,
        edge_trim_enabled: bool = True,
        line_extension_px: float = 0.0,
        cancel_check: Callable[[], bool] | None = None,
        timeout_ms: float = DEFAULT_FIBER_QUICK_GEOMETRY_TIMEOUT_MS,
    ) -> FiberQuickDiameterGeometryResult:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - dependency is required by the app
            raise RuntimeError("快速测径需要 numpy 依赖。") from exc

        started_at = perf_counter()
        deadline_at = started_at + max(50.0, float(timeout_ms)) / 1000.0
        diagnostics: dict[str, object] = {
            "skeleton_backend": FIBER_QUICK_SKELETON_BACKEND,
            "geometry_revision": FIBER_QUICK_GEOMETRY_REVISION,
        }

        def fail(code: str, message: str) -> None:
            raise FiberQuickGeometryError(
                code, message, {**diagnostics, "geometry_ms": (perf_counter() - started_at) * 1000.0}
            )

        _raise_if_cancelled(cancel_check, deadline_at)
        normalized = normalize_magic_draft_mask(mask)
        if normalized is None or not np.any(normalized):
            fail("empty_mask", "未找到目标纤维区域，请重新选择目标。")
        prepared_mask = _prepare_target_mask(normalized)
        untrimmed_mask = prepared_mask
        if edge_trim_enabled:
            prepared_mask, trimmed_pixels, border_band_px = _trim_border_connected_region(
                prepared_mask,
                positive_points=positive_points or [],
            )
        else:
            trimmed_pixels = 0
            border_band_px = 0
        # Trimming may create an artificial boundary inside the image. Record
        # exactly which strips lost foreground so those edges cannot be measured.
        clipped_sides = (False, False, False, False)
        if trimmed_pixels:
            clipped_sides = (
                bool(np.any(untrimmed_mask[:, 0])),
                bool(np.any(untrimmed_mask[0, :])),
                bool(np.any(untrimmed_mask[:, -1])),
                bool(np.any(untrimmed_mask[-1, :])),
            )
        diagnostics.update(
            edge_trim_enabled=bool(edge_trim_enabled),
            edge_trim_pixels=int(trimmed_pixels),
            edge_trim_band_px=int(border_band_px),
            clipped_sides=clipped_sides,
        )
        selected_mask = prepared_mask.astype(bool, copy=False)
        preview_polygon = [Point(point.x, point.y) for point in (preview_polygon_points or [])]
        preview_rings = [
            [Point(point.x, point.y) for point in ring]
            for ring in (preview_area_rings_points or [])
        ]
        geometry_stats: dict[str, object] = {"opened_holes": 0}
        if len(preview_polygon) < 3 and not preview_rings:
            selected_mask, preview_rings, preview_polygon, geometry_stats = magic_mask_to_geometry(
                prepared_mask,
                positive_points=positive_points or [],
                negative_points=negative_points or [],
            )
        if selected_mask is None or not np.any(selected_mask):
            fail("empty_after_cleanup", "区域清理后没有可用纤维，请补点选择更完整的目标。")
        ys, xs = np.where(selected_mask)
        diagnostics["component_area_px"] = int(len(xs))
        if len(xs) < 32:
            fail("target_too_small", "目标区域过小，无法稳定测径，请选择更完整的纤维。")
        image_h, image_w = selected_mask.shape[:2]
        image_area = max(1, image_h * image_w)
        mask_area = int(np.count_nonzero(selected_mask))
        bbox_width = int(xs.max() - xs.min() + 1)
        bbox_height = int(ys.max() - ys.min() + 1)
        diagnostics["bbox_size"] = (bbox_width, bbox_height)
        if min(bbox_width, bbox_height) < 3:
            fail("target_too_thin", "纤维的像素宽度不足，请使用更高分辨率的图像。")
        mask_area_ratio = mask_area / image_area
        # A long or diagonal fiber can occupy nearly the entire bounding box
        # while its foreground area is small. Judge actual foreground coverage.
        if mask_area_ratio >= 0.40:
            fail("target_too_large", "目标分割范围过大，请补点排除背景或相邻纤维。")

        roi_mask, roi_origin = _crop_mask_roi(selected_mask)
        touch_left = roi_origin[0] == 0 and bool(np.any(roi_mask[:, 0]))
        touch_top = roi_origin[1] == 0 and bool(np.any(roi_mask[0, :]))
        touch_right = (roi_origin[0] + roi_mask.shape[1]) >= selected_mask.shape[1] and bool(np.any(roi_mask[:, -1]))
        touch_bottom = (roi_origin[1] + roi_mask.shape[0]) >= selected_mask.shape[0] and bool(np.any(roi_mask[-1, :]))
        working_mask, _ = _resize_mask_for_geometry(roi_mask)
        scale_x = roi_mask.shape[1] / working_mask.shape[1]
        scale_y = roi_mask.shape[0] / working_mask.shape[0]

        def original_point(point: Point) -> Point:
            return Point(
                (point.x + 0.5) * scale_x - 0.5 + roi_origin[0],
                (point.y + 0.5) * scale_y - 0.5 + roi_origin[1],
            )

        _raise_if_cancelled(cancel_check, deadline_at)
        prepared_at = perf_counter()
        skeleton = _compute_skeleton(working_mask, cancel_check=cancel_check, deadline_at=deadline_at)
        skeleton_at = perf_counter()
        distance_map = cv2.distanceTransform(working_mask.astype(np.uint8), cv2.DIST_L2, 5)
        skeleton = _prune_short_spurs(
            skeleton, distance_map, cancel_check=cancel_check, deadline_at=deadline_at
        )
        local_branches = _branch_points(skeleton)
        local_endpoints = _end_points(skeleton)
        branch_points = [original_point(point) for point in local_branches]
        end_points = [original_point(point) for point in local_endpoints]
        forbidden_mask, border_seed_count = _border_forbidden_zone(
            working_mask=working_mask,
            skeleton=skeleton,
            distance_map=distance_map,
            touch_left=touch_left,
            touch_top=touch_top,
            touch_right=touch_right,
            touch_bottom=touch_bottom,
            cancel_check=cancel_check,
            deadline_at=deadline_at,
        )
        aspect_ratio = max(bbox_width, bbox_height) / max(1.0, min(bbox_width, bbox_height))
        max_candidates = 16 if aspect_ratio >= 3.0 else 12
        rejected: dict[str, int] = {}
        attempts: list[dict[str, object]] = []
        diagnostics.update(
            roi_size=(int(working_mask.shape[1]), int(working_mask.shape[0])),
            roi_scale_xy=(scale_x, scale_y),
            skeleton_pixels=int(np.count_nonzero(skeleton)),
            branch_point_count=len(branch_points),
            end_point_count=len(end_points),
            rejection_counts=rejected,
            geometry_attempts=attempts,
        )
        candidates: list[_CandidateLine] = []

        def measure(center: Point, tangent: tuple[float, float], *, strict_support: bool) -> None:
            _raise_if_cancelled(cancel_check, deadline_at)
            local_center = Point(
                (center.x - roi_origin[0] + 0.5) / scale_x - 0.5,
                (center.y - roi_origin[1] + 0.5) / scale_y - 0.5,
            )
            candidate = _measure_candidate_line(
                selected_mask=selected_mask,
                center=center,
                local_center=local_center,
                tangent=tangent,
                distance_map=distance_map,
                branch_points=branch_points,
                end_points=end_points,
                image_shape=selected_mask.shape[:2],
                scale=max(scale_x, scale_y),
                cancel_check=cancel_check,
                deadline_at=deadline_at,
                rejection_counts=rejected,
                clipped_sides=clipped_sides,
                border_band_px=border_band_px,
                strict_support=strict_support,
            )
            if candidate is not None:
                candidates.append(candidate)

        _raise_if_cancelled(cancel_check, deadline_at)
        compact_axis = _compact_mask_axis(roi_mask, diagnostics=diagnostics)

        def measure_compact_axis() -> None:
            if compact_axis is None:
                return
            center, tangent, extent = compact_axis
            for offset in (0.0, -extent * 0.1, extent * 0.1):
                point = Point(center.x + tangent[0] * offset + roi_origin[0], center.y + tangent[1] * offset + roi_origin[1])
                measure(point, tangent, strict_support=True)

        # A short fiber's skeleton may contain only a few stair-step pixels.
        # Its whole-mask axis has more directional support when the shape is
        # solid and convex; do not let that tiny skeleton overrule it.
        method = "compact_mask_axis"
        compact_attempted = False
        if compact_axis is not None and float(diagnostics.get("mask_axis_aspect", 99.0)) <= 2.5:
            compact_attempted = True
            measure_compact_axis()
            attempts.append({"method": method, "sampled": 3, "accepted": len(candidates)})
        for method, junction_factor in (("skeleton", 1.5), ("local_branch", 0.9)):
            if candidates:
                method = str(attempts[-1]["method"])
                break
            if method == "local_branch" and not local_branches:
                continue
            direction_skeleton = _exclude_junctions(
                skeleton, distance_map, local_branches, radius_factor=junction_factor
            )
            candidate_points = _select_candidate_points(
                skeleton=direction_skeleton,
                distance_map=distance_map,
                end_points=local_endpoints,
                forbidden_mask=forbidden_mask,
                max_candidates=max_candidates,
            )
            attempts.append({"method": method, "sampled": len(candidate_points)})
            for center_x, center_y in candidate_points:
                _raise_if_cancelled(cancel_check, deadline_at)
                local_center = Point(float(center_x), float(center_y))
                tangent = _estimate_tangent(working_mask, direction_skeleton, local_center, distance_map)
                if tangent is None:
                    _record_rejection(rejected, "ambiguous_direction")
                    continue
                tx, ty = tangent[0] * scale_x, tangent[1] * scale_y
                norm = hypot(tx, ty)
                measure(original_point(local_center), (tx / norm, ty / norm), strict_support=method != "skeleton")
            attempts[-1]["accepted"] = len(candidates)
            if candidates:
                break

        if not candidates and not compact_attempted:
            # Short solid fibers can have too few skeleton pixels for local PCA.
            # Only an elongated, filled shape may use its global mask axis.
            _raise_if_cancelled(cancel_check, deadline_at)
            attempts.append({"method": "compact_mask_axis", "sampled": 0, "accepted": 0})
            if compact_axis is not None:
                method = "compact_mask_axis"
                measure_compact_axis()
                attempts[-1].update(sampled=3, accepted=len(candidates))
            elif not branch_points:
                _record_rejection(rejected, "ambiguous_direction")
        if not candidates:
            code, message = _geometry_failure_reason(rejected, len(branch_points))
            fail(code, message)

        representative = _pick_representative_candidate(candidates)
        if representative is None:
            fail("low_confidence", "测量截面不够稳定，请补点选择边界更清晰的纤维段。")
        output_line = _apply_line_extension(
            representative.line,
            extension_px=float(line_extension_px),
            cancel_check=cancel_check,
            deadline_at=deadline_at,
        )

        return FiberQuickDiameterGeometryResult(
            line_px=output_line,
            confidence=max(0.0, min(1.0, representative.score)),
            status="fiber_quick",
            preview_polygon_px=preview_polygon,
            preview_area_rings_px=preview_rings,
            debug_payload={
                **diagnostics,
                "geometry_method": method,
                "candidate_count": len(candidates),
                "branch_point_count": len(branch_points),
                "end_point_count": len(local_endpoints),
                "component_area_px": mask_area,
                "opened_holes": int(geometry_stats.get("opened_holes", 0) or 0),
                "segmentation_roi_round": geometry_stats.get("segmentation_roi_round"),
                "segmentation_used_full_image": geometry_stats.get("segmentation_used_full_image"),
                "geometry_ms": (perf_counter() - started_at) * 1000.0,
                "geometry_prepare_ms": (prepared_at - started_at) * 1000.0,
                "geometry_skeleton_ms": (skeleton_at - prepared_at) * 1000.0,
                "geometry_candidates_ms": (perf_counter() - skeleton_at) * 1000.0,
                "skeleton_backend": FIBER_QUICK_SKELETON_BACKEND,
                "roi_size": (int(working_mask.shape[1]), int(working_mask.shape[0])),
                "roi_scale_xy": (scale_x, scale_y),
                "border_forbidden_seed_count": int(border_seed_count),
                "edge_trim_enabled": bool(edge_trim_enabled),
                "edge_trim_pixels": int(trimmed_pixels),
                "edge_trim_band_px": int(border_band_px),
                "line_extension_px": float(line_extension_px),
                "uncorrected_diameter_px": representative.width_px,
                "diameter_correction_px": distance(output_line.start, output_line.end) - representative.width_px,
            },
        )


@dataclass(slots=True)
class _CandidateLine:
    line: Line
    width_px: float
    score: float


def _record_rejection(counts: dict[str, int], reason: str) -> None:
    counts[reason] = counts.get(reason, 0) + 1


def _geometry_failure_reason(counts: dict[str, int], branch_count: int) -> tuple[str, str]:
    messages = {
        "incomplete_boundary": "目标在视野边缘不完整，无法取得横截面的两侧边界，请移动视野后重试。",
        "target_too_thin": "纤维的像素宽度不足，请使用更高分辨率的图像。",
        "width_mismatch": "候选截面宽度异常，可能包含交叉或粘连，请补点修正分割。",
        "unstable_width": "截面附近宽度变化过大，请选择边界更清晰的纤维段。",
        "no_boundary": "未找到完整的两侧边界，请补点修正纤维区域。",
        "ambiguous_direction": "目标方向不明确，请选择更完整的纤维段。",
    }
    if counts:
        code = max(counts, key=counts.get)
        return code, messages[code]
    if branch_count:
        return "no_usable_branch", "交叉附近没有足够的完整纤维段，请补点缩小到单根纤维。"
    return "no_usable_skeleton", "目标没有可用的测量方向，请选择更完整的纤维段。"


def _compact_mask_axis(mask, *, diagnostics=None) -> tuple[Point, tuple[float, float], float] | None:
    """Use whole-mask moments only for a solid, elongated, approximately straight shape."""
    import numpy as np

    raster = mask.astype(np.uint8)
    moments = cv2.moments(raster, binaryImage=True)
    area = moments["m00"]
    if area < 32:
        return None
    covariance = np.array([
        [moments["mu20"], moments["mu11"]],
        [moments["mu11"], moments["mu02"]],
    ]) / area
    values, vectors = np.linalg.eigh(covariance)
    if diagnostics is not None:
        diagnostics["mask_axis_ratio"] = float(values[-1] / max(values[0], 1e-6))
    if values[-1] < max(1.0, values[0] * 1.35):
        return None
    contours, _ = cv2.findContours(raster, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 1:
        return None
    rectangle = cv2.minAreaRect(contours[0])
    _, (width, height), _ = rectangle
    if diagnostics is not None:
        diagnostics["mask_axis_aspect"] = float(max(width, height) / max(1.0, min(width, height)))
    fill_ratio = area / ((width + 1.0) * (height + 1.0))
    concavity = cv2.contourArea(cv2.convexHull(contours[0])) - cv2.contourArea(contours[0])
    # Stair-step digital edges have small concavities even for a rectangle.
    # Permit that pixel-scale loss, but not a concave cross, T or bent fiber.
    raster_tolerance = cv2.arcLength(contours[0], True) * 0.6
    if diagnostics is not None:
        diagnostics.update(mask_axis_fill_ratio=fill_ratio, mask_concavity_px=concavity)
    if fill_ratio < 0.78 or concavity > max(2.0, raster_tolerance):
        return None
    tangent = vectors[:, -1]
    box = cv2.boxPoints(rectangle)
    sides = np.roll(box, -1, axis=0) - box
    longest = sides[int(np.argmax(np.sum(sides * sides, axis=1)))]
    alignment = abs(float(tangent @ longest)) / max(float(np.linalg.norm(longest)), 1e-6)
    if diagnostics is not None:
        diagnostics["mask_axis_agreement"] = alignment
    # At very low resolution the moments and enclosing box can disagree by
    # tens of degrees. That is insufficient evidence to recover a short fiber.
    if alignment < 0.985:
        return None
    return (
        Point(moments["m10"] / area, moments["m01"] / area),
        (float(tangent[0]), float(tangent[1])),
        float(max(width, height)),
    )


def _prepare_target_mask(mask):
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency is required by the app
        raise RuntimeError("快速测径需要 numpy 依赖。") from exc

    # Give morphology explicit background outside the image. Its default
    # border extrapolation can close a small gap to the frame and falsely make
    # an otherwise complete fiber appear clipped. Crop back afterward so a
    # fiber that really crosses the source boundary still touches that edge.
    padding = 6
    working = np.pad(np.asarray(mask, dtype=np.uint8), padding, mode="constant")
    working = cv2.morphologyEx(working, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
    working = cv2.morphologyEx(working, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    working = working[padding:-padding, padding:-padding].copy()
    filled = fill_magic_draft_internal_holes(working.astype(bool))
    if filled is not None:
        working = filled.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(working, connectivity=8)
    if num_labels <= 1:
        return working.astype(bool)
    best_label = max(range(1, num_labels), key=lambda label: int(stats[label, cv2.CC_STAT_AREA]))
    return (labels == best_label)


def _trim_border_connected_region(mask, *, positive_points: list[Point]):
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency is required by the app
        raise RuntimeError("快速测径需要 numpy 依赖。") from exc

    working = np.asarray(mask, dtype=bool).copy()
    if not np.any(working):
        return working, 0, 0
    image_h, image_w = working.shape[:2]
    band_px = int(max(4, min(12, round(max(image_h, image_w) * 0.005))))
    border_band = np.zeros_like(working, dtype=bool)
    # Proximity alone does not make a fiber incomplete. Only trim sides that
    # actually touch the source image boundary.
    if np.any(working[0, :]):
        border_band[:band_px, :] = True
    if np.any(working[-1, :]):
        border_band[-band_px:, :] = True
    if np.any(working[:, 0]):
        border_band[:, :band_px] = True
    if np.any(working[:, -1]):
        border_band[:, -band_px:] = True
    touched = bool(np.any(working & border_band))
    if not touched:
        return working, 0, band_px
    removed = int(np.count_nonzero(working & border_band))
    trimmed = working.copy()
    trimmed[border_band] = False
    if not np.any(trimmed):
        return trimmed, removed, band_px
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(trimmed.astype("uint8"), connectivity=8)
    if num_labels <= 1:
        return trimmed, removed, band_px
    target_center = None
    if positive_points:
        total_x = sum(point.x for point in positive_points)
        total_y = sum(point.y for point in positive_points)
        target_center = (total_x / len(positive_points), total_y / len(positive_points))
    best_label = 1
    best_key = None
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        centroid_x, centroid_y = centroids[label]
        dist = 0.0
        if target_center is not None:
            dist = ((float(centroid_x) - target_center[0]) ** 2 + (float(centroid_y) - target_center[1]) ** 2) ** 0.5
        key = (dist, -area)
        if best_key is None or key < best_key:
            best_key = key
            best_label = label
    return labels == best_label, removed, band_px


_SKELETON_OFFSETS = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))


def _skeleton_neighbors(skeleton):
    """Eight-connected graph, without diagonal shortcuts across existing corners."""
    import numpy as np

    padded = np.pad(skeleton.astype(bool), 1)
    neighbors = [
        padded[:-2, 1:-1], padded[:-2, 2:], padded[1:-1, 2:], padded[2:, 2:],
        padded[2:, 1:-1], padded[2:, :-2], padded[1:-1, :-2], padded[:-2, :-2],
    ]
    # A diagonal is redundant if an orthogonal two-edge path already connects
    # these pixels. Counting it creates false junctions on stair-step curves.
    for index in (1, 3, 5, 7):
        neighbors[index] = neighbors[index] & ~(neighbors[index - 1] | neighbors[(index + 1) % 8])
    return np.asarray(neighbors)


def _prune_short_spurs(skeleton, distance_map, *, cancel_check=None, deadline_at=None):
    """Remove terminal branches shorter than the local diameter at their junction.

    Rotated flat end caps often skeletonize into two short spurs. Retaining them
    both removes useful shaft sections near a false junction and samples cap
    widths. Never remove an unbranched fiber, regardless of its length.
    """
    import numpy as np

    usable = skeleton.copy()
    for _ in range(8):
        _raise_if_cancelled(cancel_check, deadline_at)
        neighbors = _skeleton_neighbors(usable)
        degree = neighbors.sum(axis=0)
        junctions = usable & (degree > 2)
        if not np.any(junctions):
            break
        max_length = 2.0 * float(distance_map[junctions].max())
        removed: set[tuple[int, int]] = set()
        for y, x in zip(*np.where(usable & (degree == 1))):
            _raise_if_cancelled(cancel_check, deadline_at)
            current = (int(y), int(x))
            previous = None
            path = []
            length = 0.0
            while length <= max_length:
                if len(path) % 128 == 0:
                    _raise_if_cancelled(cancel_check, deadline_at)
                if degree[current] > 2:
                    if length <= 2.0 * float(distance_map[current]):
                        removed.update(path)
                    break
                path.append(current)
                next_pixels = [
                    (current[0] + _SKELETON_OFFSETS[index][0], current[1] + _SKELETON_OFFSETS[index][1])
                    for index in np.flatnonzero(neighbors[:, current[0], current[1]])
                ]
                next_pixels = [pixel for pixel in next_pixels if pixel != previous]
                if len(next_pixels) != 1:
                    break
                next_pixel = next_pixels[0]
                length += hypot(next_pixel[0] - current[0], next_pixel[1] - current[1])
                previous, current = current, next_pixel
        if not removed:
            break
        # All arms of a short junction may be terminal. Do not collapse the
        # whole structure into its central knot by labelling every arm a spur.
        if int(np.count_nonzero(usable)) - len(removed) < max(6, ceil(max_length)):
            break
        ys, xs = zip(*removed)
        usable[ys, xs] = False
    return usable


def _exclude_junctions(skeleton, distance_map, branch_points, *, radius_factor: float = 1.5):
    """Remove a width-dependent neighborhood before sampling or fitting directions."""
    import numpy as np

    usable = skeleton.astype(np.uint8)
    for point in branch_points:
        radius = max(2, int(ceil(_sample_image(distance_map, point) * radius_factor)))
        _draw_block_circle(usable, point, radius=radius, value=0)
    return usable.astype(bool)


def _select_candidate_points(
    *,
    skeleton,
    distance_map,
    end_points: list[Point],
    forbidden_mask,
    max_candidates: int,
) -> list[tuple[int, int]]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency is required by the app
        raise RuntimeError("快速测径需要 numpy 依赖。") from exc

    source = skeleton.astype(bool)
    if not np.any(source):
        return []
    candidate_map = np.logical_and(source, distance_map >= 1.5)
    if not np.any(candidate_map):
        candidate_map = source

    working = candidate_map.astype(np.uint8)
    if forbidden_mask is not None and np.any(forbidden_mask):
        working[forbidden_mask.astype(bool)] = 0
    for point in end_points:
        _draw_block_circle(working, point, radius=3, value=0)

    ys, xs = np.where(working > 0)
    if len(xs) == 0:
        return []
    coords = np.column_stack((xs, ys)).astype(np.float64)
    # Spread samples over the usable branches instead of preferring the thickest
    # pixels (which systematically favors junctions and swollen fiber regions).
    index = int(np.argmin(np.sum((coords - coords.mean(axis=0)) ** 2, axis=1)))
    nearest_squared = np.full(len(coords), np.inf)
    chosen: list[tuple[int, int]] = []
    for _ in range(min(max_candidates, len(coords))):
        point = coords[index]
        chosen.append((int(point[0]), int(point[1])))
        nearest_squared = np.minimum(nearest_squared, np.sum((coords - point) ** 2, axis=1))
        index = int(np.argmax(nearest_squared))
        if nearest_squared[index] < 25.0:
            break
    return chosen


def _estimate_tangent(selected_mask, skeleton, center: Point, distance_map):
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency is required by the app
        raise RuntimeError("快速测径需要 numpy 依赖。") from exc

    radius = max(8, int(round(_sample_image(distance_map, center) * 1.5)))
    min_x = max(0, int(round(center.x)) - radius)
    max_x = min(selected_mask.shape[1], int(round(center.x)) + radius + 1)
    min_y = max(0, int(round(center.y)) - radius)
    max_y = min(selected_mask.shape[0], int(round(center.y)) + radius + 1)
    source = skeleton[min_y:max_y, min_x:max_x]
    # Nearby parallel branches and the other arm of a junction must not rotate
    # this section's tangent. Fit only the component containing the candidate.
    _, labels = cv2.connectedComponents(source.astype(np.uint8), connectivity=8)
    label = labels[int(round(center.y)) - min_y, int(round(center.x)) - min_x]
    if label == 0:
        return None
    source = labels == label
    ys, xs = np.where(source)
    if len(xs) < 3:
        source = selected_mask[min_y:max_y, min_x:max_x]
        ys, xs = np.where(source)
    if len(xs) < 6:
        return None
    coords = np.column_stack((xs + min_x, ys + min_y)).astype(np.float32)
    mean = coords.mean(axis=0)
    centered = coords - mean
    cov = centered.T @ centered / max(1, len(coords) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    if eigenvalues[-1] <= max(1e-6, float(eigenvalues[0]) * 2.0):
        return None
    tangent = eigenvectors[:, int(np.argmax(eigenvalues))]
    norm = float(np.hypot(tangent[0], tangent[1]))
    if norm <= 1e-6:
        return None
    return float(tangent[0] / norm), float(tangent[1] / norm)


def _measure_candidate_line(
    *,
    selected_mask,
    center: Point,
    local_center: Point,
    tangent: tuple[float, float],
    distance_map,
    branch_points: list[Point],
    end_points: list[Point],
    image_shape: tuple[int, int],
    scale: float,
    cancel_check: Callable[[], bool] | None,
    deadline_at: float | None,
    rejection_counts: dict[str, int] | None = None,
    clipped_sides: tuple[bool, bool, bool, bool] = (False, False, False, False),
    border_band_px: int = 0,
    strict_support: bool = False,
) -> _CandidateLine | None:
    def reject(reason: str) -> None:
        if rejection_counts is not None:
            _record_rejection(rejection_counts, reason)

    normal = (-tangent[1], tangent[0])
    line, hit_image_border = _measure_line(selected_mask, center, normal, cancel_check=cancel_check, deadline_at=deadline_at)
    if line is None or hit_image_border:
        reject("incomplete_boundary" if hit_image_border else "no_boundary")
        return None
    if _line_touches_trimmed_side(line, image_shape, clipped_sides, border_band_px):
        reject("incomplete_boundary")
        return None
    line_length_px = distance(line.start, line.end)
    if line_length_px < 3.0:
        reject("target_too_thin")
        return None
    radius = max(2.0, _sample_image(distance_map, local_center) * scale)
    if line_length_px > (2.0 * radius * 1.4):
        reject("width_mismatch")
        return None
    border_clearance = min(
        center.x,
        center.y,
        image_shape[1] - 1 - center.x,
        image_shape[0] - 1 - center.y,
    )
    offset = max(2.0, min(radius * 0.8, 6.0))
    if strict_support:
        offset = max(2.0, min(radius * 0.6, 12.0))
    offset_a = Point(center.x + (tangent[0] * offset), center.y + (tangent[1] * offset))
    offset_b = Point(center.x - (tangent[0] * offset), center.y - (tangent[1] * offset))
    sample_widths = [line_length_px]
    for offset_center in (offset_a, offset_b):
        _raise_if_cancelled(cancel_check, deadline_at)
        offset_line, offset_hit_border = _measure_line(selected_mask, offset_center, normal, cancel_check=cancel_check, deadline_at=deadline_at)
        if (
            offset_line is not None and not offset_hit_border
            and not _line_touches_trimmed_side(offset_line, image_shape, clipped_sides, border_band_px)
        ):
            sample_widths.append(distance(offset_line.start, offset_line.end))
    if strict_support and (
        len(sample_widths) < 3
        or max(sample_widths) - min(sample_widths) > max(2.0, line_length_px * 0.15)
    ):
        reject("unstable_width")
        return None
    stability = 1.0 - min(_coefficient_of_variation(sample_widths), 1.0)
    symmetry = min(distance(center, line.start), distance(center, line.end)) / max(distance(center, line.start), distance(center, line.end), 1e-6)
    branch_clearance = min((distance(center, branch) for branch in branch_points), default=999.0)
    branch_score = min(branch_clearance / 24.0, 1.0)
    radius_score = min(radius / 18.0, 1.0)
    border_score = min(border_clearance / max(12.0, radius * 2.5), 1.0)
    score = (0.28 * symmetry) + (0.24 * stability) + (0.20 * branch_score) + (0.10 * radius_score) + (0.18 * border_score)
    # End caps can have a stable width but a bent skeleton. Prefer shaft
    # sections with enough directional support on both sides of the sample.
    end_clearance = min((distance(center, endpoint) for endpoint in end_points), default=999.0)
    end_score = min(end_clearance / max(8.0, radius * 2.0), 1.0)
    score *= 0.7 + (0.3 * end_score)
    return _CandidateLine(line=line, width_px=line_length_px, score=float(score))


def _line_touches_trimmed_side(line, shape, clipped_sides, band_px) -> bool:
    if not any(clipped_sides):
        return False
    height, width = shape
    left, top, right, bottom = clipped_sides
    margin = band_px + 0.5
    return any(
        (left and point.x <= margin) or (top and point.y <= margin)
        or (right and point.x >= width - 1 - margin)
        or (bottom and point.y >= height - 1 - margin)
        for point in (line.start, line.end)
    )


def _measure_line(
    selected_mask,
    center: Point,
    normal: tuple[float, float],
    *,
    cancel_check: Callable[[], bool] | None = None,
    deadline_at: float | None = None,
) -> tuple[Line | None, bool]:
    left, left_hit_border = _walk_to_boundary(selected_mask, center, normal, direction_sign=-1.0, cancel_check=cancel_check, deadline_at=deadline_at)
    right, right_hit_border = _walk_to_boundary(selected_mask, center, normal, direction_sign=1.0, cancel_check=cancel_check, deadline_at=deadline_at)
    if left is None or right is None:
        return None, left_hit_border or right_hit_border
    return Line(start=left, end=right), left_hit_border or right_hit_border


def _pick_representative_candidate(candidates: list[_CandidateLine]) -> _CandidateLine | None:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency is required by the app
        raise RuntimeError("快速测径需要 numpy 依赖。") from exc

    candidates = [candidate for candidate in candidates if candidate.score >= 0.22]
    if not candidates:
        return None
    widths = np.array([candidate.width_px for candidate in candidates], dtype=np.float64)
    median = float(np.median(widths))
    tolerance = max(2.0, median * 0.18)
    filtered = [candidate for candidate in candidates if abs(candidate.width_px - median) <= tolerance]
    if not filtered:
        filtered = candidates
    ranked = sorted(
        filtered,
        # Sub-micropixel floating-point noise must not override confidence or
        # the deterministic center-first sampling order for equivalent widths.
        key=lambda candidate: (round(abs(candidate.width_px - median), 6), -round(candidate.score, 6)),
    )
    return ranked[0]


def _apply_line_extension(
    line: Line,
    *,
    extension_px: float,
    cancel_check: Callable[[], bool] | None,
    deadline_at: float | None,
) -> Line:
    """Apply a signed correction per endpoint, in original image pixels.

    The measured line already reaches the mask boundary. Correction must be
    geometric: searching inside that mask again would prevent it from growing.
    """
    _raise_if_cancelled(cancel_check, deadline_at)
    extension = float(extension_px)
    if not isfinite(extension):
        raise ValueError("快速测径扩展像素必须是有限数值。")
    if abs(extension) <= 1e-6:
        return line
    length = distance(line.start, line.end)
    if length <= 1e-6:
        return line
    unit_x = (line.end.x - line.start.x) / length
    unit_y = (line.end.y - line.start.y) / length
    # Preserve the midpoint and direction; keep the existing 2 px minimum so
    # an excessive negative correction cannot reverse the endpoints.
    if extension < 0:
        extension = max(extension, min(0.0, 1.0 - length / 2.0))
    start = Point(line.start.x - (unit_x * extension), line.start.y - (unit_y * extension))
    end = Point(line.end.x + (unit_x * extension), line.end.y + (unit_y * extension))
    return Line(start=start, end=end)


def _coefficient_of_variation(values: list[float]) -> float:
    if not values:
        return 1.0
    mean = sum(values) / len(values)
    if mean <= 1e-6:
        return 1.0
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return (variance ** 0.5) / mean


def _walk_to_boundary(
    selected_mask,
    center: Point,
    axis: tuple[float, float],
    *,
    direction_sign: float,
    step: float = 1.0,
    cancel_check: Callable[[], bool] | None = None,
    deadline_at: float | None = None,
) -> tuple[Point | None, bool]:
    import numpy as np

    height, width = selected_mask.shape[:2]
    norm = hypot(*axis)
    if not isfinite(norm) or norm <= 1e-12 or not isfinite(step) or step <= 0:
        return None, False
    dir_x = axis[0] / norm * direction_sign
    dir_y = axis[1] / norm * direction_sign
    # A unit ray must leave the image within its diagonal. Do not interpret a
    # fixed iteration cap as a found boundary (the old 480-step limit did so).
    max_steps = int(ceil(hypot(width, height) / step)) + 2
    last_inside = None
    for start in range(0, max_steps, 128):
        _raise_if_cancelled(cancel_check, deadline_at)
        positions = np.arange(start, min(start + 128, max_steps), dtype=np.float64) * step
        xs = center.x + positions * dir_x
        ys = center.y + positions * dir_y
        ix = np.rint(xs).astype(np.int64)
        iy = np.rint(ys).astype(np.int64)
        in_image = (ix >= 0) & (iy >= 0) & (ix < width) & (iy < height)
        inside = np.zeros(len(positions), dtype=bool)
        inside[in_image] = selected_mask[iy[in_image], ix[in_image]]
        exits = np.flatnonzero(~inside)
        if len(exits):
            index = int(exits[0])
            if index:
                last_inside = Point(float(xs[index - 1]), float(ys[index - 1]))
            return last_inside, not bool(in_image[index])
        last_inside = Point(float(xs[-1]), float(ys[-1]))
    return None, True


def _sample_image(image, point: Point) -> float:
    height, width = image.shape[:2]
    x = max(0, min(width - 1, int(round(point.x))))
    y = max(0, min(height - 1, int(round(point.y))))
    return float(image[y, x])


def _draw_block_circle(image, point: Point, *, radius: int, value: int) -> None:
    center = (int(round(point.x)), int(round(point.y)))
    if getattr(image, "dtype", None) is not None and image.dtype == bool:
        raster = image.astype("uint8", copy=True)
        cv2.circle(raster, center, radius, int(value), thickness=-1)
        image[:] = raster.astype(bool)
        return
    cv2.circle(image, center, radius, int(value), thickness=-1)


def _branch_points(skeleton, *, origin: tuple[int, int] = (0, 0), scale: float = 1.0) -> list[Point]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency is required by the app
        raise RuntimeError("快速测径需要 numpy 依赖。") from exc

    if skeleton is None or not np.any(skeleton):
        return []
    skeleton_u8 = skeleton.astype(np.uint8)
    neighbors = _skeleton_neighbors(skeleton).sum(axis=0)
    ys, xs = np.where((skeleton_u8 > 0) & (neighbors > 2))
    return [Point(float((x * scale) + origin[0]), float((y * scale) + origin[1])) for y, x in zip(ys, xs, strict=False)]


def _end_points(skeleton, *, origin: tuple[int, int] = (0, 0), scale: float = 1.0) -> list[Point]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency is required by the app
        raise RuntimeError("快速测径需要 numpy 依赖。") from exc

    if skeleton is None or not np.any(skeleton):
        return []
    skeleton_u8 = skeleton.astype(np.uint8)
    neighbors = _skeleton_neighbors(skeleton).sum(axis=0)
    ys, xs = np.where((skeleton_u8 > 0) & (neighbors == 1))
    return [Point(float((x * scale) + origin[0]), float((y * scale) + origin[1])) for y, x in zip(ys, xs, strict=False)]


def _compute_skeleton(mask, *, cancel_check: Callable[[], bool] | None = None, deadline_at: float | None = None):
    _raise_if_cancelled(cancel_check, deadline_at)
    skeletonize = prepare_fiber_quick_geometry_backend()
    _raise_if_cancelled(cancel_check, deadline_at)
    skeleton = skeletonize(mask.astype(bool), method="zhang")
    _raise_if_cancelled(cancel_check, deadline_at)
    return skeleton


def _crop_mask_roi(mask, *, padding: int = 12):
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency is required by the app
        raise RuntimeError("快速测径需要 numpy 依赖。") from exc

    ys, xs = np.where(mask)
    min_x = max(0, int(xs.min()) - padding)
    max_x = min(mask.shape[1], int(xs.max()) + padding + 1)
    min_y = max(0, int(ys.min()) - padding)
    max_y = min(mask.shape[0], int(ys.max()) + padding + 1)
    return mask[min_y:max_y, min_x:max_x].copy(), (min_x, min_y)


def _resize_mask_for_geometry(mask, *, max_long_side: int = 640):
    height, width = mask.shape[:2]
    long_side = max(height, width)
    if long_side <= max_long_side:
        return mask.copy(), 1.0
    scale = long_side / float(max_long_side)
    average_width = float(mask.sum()) / long_side
    if average_width / scale < 4.0:
        # A long thin ROI often has few total pixels. Preserve its width instead
        # of squeezing a sound fiber into one pixel just to meet a side limit.
        area_scale = max(1.0, (mask.size / (max_long_side ** 2)) ** 0.5)
        scale = max(area_scale, min(scale, max(1.0, average_width / 4.0)))
    target_w = max(1, int(round(width / scale)))
    target_h = max(1, int(round(height / scale)))
    resized = cv2.resize(mask.astype("uint8"), (target_w, target_h), interpolation=cv2.INTER_NEAREST_EXACT) > 0
    return resized, float(scale)


def _border_forbidden_zone(
    *,
    working_mask,
    skeleton,
    distance_map,
    touch_left: bool,
    touch_top: bool,
    touch_right: bool,
    touch_bottom: bool,
    cancel_check: Callable[[], bool] | None,
    deadline_at: float | None,
):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("快速测径需要 numpy 依赖。") from exc

    border_touch = np.zeros_like(working_mask, dtype=bool)
    if touch_left:
        border_touch[:, 0] = working_mask[:, 0]
    if touch_top:
        border_touch[0, :] = working_mask[0, :]
    if touch_right:
        border_touch[:, -1] = working_mask[:, -1]
    if touch_bottom:
        border_touch[-1, :] = working_mask[-1, :]
    if not np.any(border_touch) or not np.any(skeleton):
        return np.zeros_like(working_mask, dtype=bool), 0

    _raise_if_cancelled(cancel_check, deadline_at)
    dist_to_border = cv2.distanceTransform((~border_touch).astype("uint8"), cv2.DIST_L2, 5)
    seed_map = skeleton.astype(bool) & (dist_to_border <= 6.0)
    seed_count = int(np.count_nonzero(seed_map))
    if seed_count == 0:
        return np.zeros_like(working_mask, dtype=bool), 0

    seed_radii = distance_map[seed_map]
    median_radius = float(np.median(seed_radii)) if seed_radii.size else 0.0
    expand_steps = int(round(min(48.0, max(16.0, 3.0 * median_radius))))
    forbidden = seed_map.astype("uint8")
    skeleton_u8 = skeleton.astype("uint8")
    kernel = np.ones((3, 3), dtype=np.uint8)
    for _ in range(max(1, expand_steps)):
        _raise_if_cancelled(cancel_check, deadline_at)
        forbidden = cv2.dilate(forbidden, kernel, iterations=1)
        forbidden = cv2.bitwise_and(forbidden, skeleton_u8)
    return forbidden.astype(bool), seed_count


def _effective_skeleton_length(skeleton, *, forbidden_mask) -> int:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("快速测径需要 numpy 依赖。") from exc
    if skeleton is None or not np.any(skeleton):
        return 0
    if forbidden_mask is None or not np.any(forbidden_mask):
        return int(np.count_nonzero(skeleton))
    return int(np.count_nonzero(skeleton.astype(bool) & (~forbidden_mask.astype(bool))))


def _raise_if_cancelled(cancel_check: Callable[[], bool] | None, deadline_at: float | None = None) -> None:
    if cancel_check is not None and cancel_check():
        raise RuntimeError("请求已取消。")
    if deadline_at is not None and perf_counter() >= deadline_at:
        raise RuntimeError("直径计算超时。")
