from __future__ import annotations

from dataclasses import dataclass

import cv2

from fdm.geometry import Line, Point, distance
from fdm.services.prompt_segmentation import (
    fill_magic_draft_internal_holes,
    magic_mask_to_geometry,
    normalize_magic_draft_mask,
)


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
    ) -> FiberQuickDiameterGeometryResult:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - dependency is required by the app
            raise RuntimeError("快速测径需要 numpy 依赖。") from exc

        normalized = normalize_magic_draft_mask(mask)
        if normalized is None or not np.any(normalized):
            raise RuntimeError("未找到目标纤维区域。")
        prepared_mask = _prepare_target_mask(normalized)
        selected_mask, preview_rings, preview_polygon, geometry_stats = magic_mask_to_geometry(
            prepared_mask,
            positive_points=positive_points or [],
            negative_points=negative_points or [],
        )
        if selected_mask is None or not np.any(selected_mask):
            raise RuntimeError("未找到目标纤维区域。")
        ys, xs = np.where(selected_mask)
        if len(xs) < 64:
            raise RuntimeError("未找到可靠直径线。")
        bbox_width = int(xs.max() - xs.min() + 1)
        bbox_height = int(ys.max() - ys.min() + 1)
        if min(bbox_width, bbox_height) < 6:
            raise RuntimeError("未找到可靠直径线。")

        skeleton = _zhang_suen_thinning(selected_mask)
        distance_map = cv2.distanceTransform(selected_mask.astype(np.uint8), cv2.DIST_L2, 5)
        branch_points = _branch_points(skeleton)
        end_points = _end_points(skeleton)
        candidate_points = _select_candidate_points(
            selected_mask=selected_mask,
            skeleton=skeleton,
            distance_map=distance_map,
            branch_points=branch_points,
            end_points=end_points,
        )
        if not candidate_points:
            raise RuntimeError("未找到可靠直径线。")

        candidates: list[_CandidateLine] = []
        for center_x, center_y in candidate_points:
            center = Point(float(center_x), float(center_y))
            tangent = _estimate_tangent(selected_mask, skeleton, center, distance_map)
            if tangent is None:
                continue
            candidate = _measure_candidate_line(
                selected_mask=selected_mask,
                center=center,
                tangent=tangent,
                distance_map=distance_map,
                branch_points=branch_points,
            )
            if candidate is not None:
                candidates.append(candidate)
        if not candidates:
            raise RuntimeError("未找到可靠直径线。")

        representative = _pick_representative_candidate(candidates)
        if representative is None:
            raise RuntimeError("未找到可靠直径线。")

        return FiberQuickDiameterGeometryResult(
            line_px=representative.line,
            confidence=max(0.0, min(1.0, representative.score)),
            status="fiber_quick",
            preview_polygon_px=preview_polygon,
            preview_area_rings_px=preview_rings,
            debug_payload={
                "candidate_count": len(candidates),
                "branch_point_count": len(branch_points),
                "end_point_count": len(end_points),
                "component_area_px": int(np.count_nonzero(selected_mask)),
                "opened_holes": int(geometry_stats.get("opened_holes", 0) or 0),
            },
        )


@dataclass(slots=True)
class _CandidateLine:
    line: Line
    width_px: float
    score: float


def _prepare_target_mask(mask):
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency is required by the app
        raise RuntimeError("快速测径需要 numpy 依赖。") from exc

    working = np.asarray(mask, dtype=np.uint8).copy()
    working = cv2.morphologyEx(working, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
    working = cv2.morphologyEx(working, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    filled = fill_magic_draft_internal_holes(working.astype(bool))
    if filled is not None:
        working = filled.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(working, connectivity=8)
    if num_labels <= 1:
        return working.astype(bool)
    best_label = max(range(1, num_labels), key=lambda label: int(stats[label, cv2.CC_STAT_AREA]))
    return (labels == best_label)


def _select_candidate_points(*, selected_mask, skeleton, distance_map, branch_points: list[Point], end_points: list[Point]) -> list[tuple[int, int]]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency is required by the app
        raise RuntimeError("快速测径需要 numpy 依赖。") from exc

    source = skeleton.astype(bool)
    if not np.any(source):
        source = selected_mask.astype(bool)
    candidate_map = np.logical_and(source, distance_map >= 1.5)
    if not np.any(candidate_map):
        candidate_map = source

    for point in [*branch_points, *end_points]:
        _draw_block_circle(candidate_map, point, radius=4 if point in branch_points else 3, value=0)

    ys, xs = np.where(candidate_map)
    if len(xs) == 0:
        return []
    scored = sorted(
        ((float(distance_map[y, x]), int(x), int(y)) for x, y in zip(xs, ys, strict=False)),
        reverse=True,
    )
    chosen: list[tuple[int, int]] = []
    for _score, x, y in scored:
        if any(abs(x - prev_x) <= 4 and abs(y - prev_y) <= 4 for prev_x, prev_y in chosen):
            continue
        chosen.append((x, y))
        if len(chosen) >= 48:
            break
    return chosen


def _estimate_tangent(selected_mask, skeleton, center: Point, distance_map):
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency is required by the app
        raise RuntimeError("快速测径需要 numpy 依赖。") from exc

    radius = max(6, int(round(_sample_image(distance_map, center) * 3.0)))
    min_x = max(0, int(round(center.x)) - radius)
    max_x = min(selected_mask.shape[1], int(round(center.x)) + radius + 1)
    min_y = max(0, int(round(center.y)) - radius)
    max_y = min(selected_mask.shape[0], int(round(center.y)) + radius + 1)
    source = skeleton[min_y:max_y, min_x:max_x]
    ys, xs = np.where(source)
    if len(xs) < 6:
        source = selected_mask[min_y:max_y, min_x:max_x]
        ys, xs = np.where(source)
    if len(xs) < 6:
        return None
    coords = np.column_stack((xs + min_x, ys + min_y)).astype(np.float32)
    mean = coords.mean(axis=0)
    centered = coords - mean
    cov = centered.T @ centered / max(1, len(coords) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    tangent = eigenvectors[:, int(np.argmax(eigenvalues))]
    norm = float(np.hypot(tangent[0], tangent[1]))
    if norm <= 1e-6:
        return None
    return float(tangent[0] / norm), float(tangent[1] / norm)


def _measure_candidate_line(*, selected_mask, center: Point, tangent: tuple[float, float], distance_map, branch_points: list[Point]) -> _CandidateLine | None:
    normal = (-tangent[1], tangent[0])
    line = _measure_line(selected_mask, center, normal)
    if line is None:
        return None
    line_length_px = distance(line.start, line.end)
    if line_length_px < 3.0:
        return None
    radius = max(2.0, _sample_image(distance_map, center))
    offset = max(2.0, min(radius * 0.8, 6.0))
    offset_a = Point(center.x + (tangent[0] * offset), center.y + (tangent[1] * offset))
    offset_b = Point(center.x - (tangent[0] * offset), center.y - (tangent[1] * offset))
    sample_widths = [line_length_px]
    for offset_center in (offset_a, offset_b):
        offset_line = _measure_line(selected_mask, offset_center, normal)
        if offset_line is not None:
            sample_widths.append(distance(offset_line.start, offset_line.end))
    stability = 1.0 - min(_coefficient_of_variation(sample_widths), 1.0)
    symmetry = min(distance(center, line.start), distance(center, line.end)) / max(distance(center, line.start), distance(center, line.end), 1e-6)
    branch_clearance = min((distance(center, branch) for branch in branch_points), default=999.0)
    branch_score = min(branch_clearance / 24.0, 1.0)
    radius_score = min(radius / 18.0, 1.0)
    score = (0.35 * symmetry) + (0.25 * stability) + (0.25 * branch_score) + (0.15 * radius_score)
    return _CandidateLine(line=line, width_px=line_length_px, score=float(score))


def _measure_line(selected_mask, center: Point, normal: tuple[float, float]) -> Line | None:
    left = _walk_to_boundary(selected_mask, center, normal, direction_sign=-1.0)
    right = _walk_to_boundary(selected_mask, center, normal, direction_sign=1.0)
    if left is None or right is None:
        return None
    return Line(start=left, end=right)


def _pick_representative_candidate(candidates: list[_CandidateLine]) -> _CandidateLine | None:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency is required by the app
        raise RuntimeError("快速测径需要 numpy 依赖。") from exc

    if not candidates:
        return None
    widths = np.array([candidate.width_px for candidate in candidates], dtype=np.float32)
    median = float(np.median(widths))
    tolerance = max(2.0, median * 0.18)
    filtered = [candidate for candidate in candidates if abs(candidate.width_px - median) <= tolerance]
    if not filtered:
        filtered = candidates
    ranked = sorted(
        filtered,
        key=lambda candidate: (abs(candidate.width_px - median), -candidate.score),
    )
    best = ranked[0]
    if best.score < 0.22:
        return None
    return best


def _coefficient_of_variation(values: list[float]) -> float:
    if not values:
        return 1.0
    mean = sum(values) / len(values)
    if mean <= 1e-6:
        return 1.0
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return (variance ** 0.5) / mean


def _walk_to_boundary(selected_mask, center: Point, axis: tuple[float, float], *, direction_sign: float, step: float = 0.5) -> Point | None:
    height, width = selected_mask.shape[:2]
    last_inside = None
    x = float(center.x)
    y = float(center.y)
    for _ in range(480):
        ix = int(round(x))
        iy = int(round(y))
        if ix < 0 or iy < 0 or ix >= width or iy >= height:
            break
        if not bool(selected_mask[iy, ix]):
            break
        last_inside = Point(x, y)
        x += axis[0] * step * direction_sign
        y += axis[1] * step * direction_sign
    return last_inside


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


def _branch_points(skeleton) -> list[Point]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency is required by the app
        raise RuntimeError("快速测径需要 numpy 依赖。") from exc

    if skeleton is None or not np.any(skeleton):
        return []
    padded = np.pad(skeleton.astype(np.uint8), 1, mode="constant")
    branches: list[Point] = []
    for y in range(1, padded.shape[0] - 1):
        for x in range(1, padded.shape[1] - 1):
            if padded[y, x] == 0:
                continue
            neighborhood = padded[y - 1 : y + 2, x - 1 : x + 2]
            neighbors = int(neighborhood.sum()) - 1
            if neighbors > 2:
                branches.append(Point(float(x - 1), float(y - 1)))
    return branches


def _end_points(skeleton) -> list[Point]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency is required by the app
        raise RuntimeError("快速测径需要 numpy 依赖。") from exc

    if skeleton is None or not np.any(skeleton):
        return []
    padded = np.pad(skeleton.astype(np.uint8), 1, mode="constant")
    endpoints: list[Point] = []
    for y in range(1, padded.shape[0] - 1):
        for x in range(1, padded.shape[1] - 1):
            if padded[y, x] == 0:
                continue
            neighborhood = padded[y - 1 : y + 2, x - 1 : x + 2]
            neighbors = int(neighborhood.sum()) - 1
            if neighbors == 1:
                endpoints.append(Point(float(x - 1), float(y - 1)))
    return endpoints


def _zhang_suen_thinning(mask):
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency is required by the app
        raise RuntimeError("快速测径需要 numpy 依赖。") from exc

    image = mask.astype(np.uint8).copy()
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            to_remove: list[tuple[int, int]] = []
            for y in range(1, image.shape[0] - 1):
                for x in range(1, image.shape[1] - 1):
                    if image[y, x] == 0:
                        continue
                    p2 = image[y - 1, x]
                    p3 = image[y - 1, x + 1]
                    p4 = image[y, x + 1]
                    p5 = image[y + 1, x + 1]
                    p6 = image[y + 1, x]
                    p7 = image[y + 1, x - 1]
                    p8 = image[y, x - 1]
                    p9 = image[y - 1, x - 1]
                    neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
                    if neighbors < 2 or neighbors > 6:
                        continue
                    transitions = int(
                        (p2 == 0 and p3 == 1)
                        + (p3 == 0 and p4 == 1)
                        + (p4 == 0 and p5 == 1)
                        + (p5 == 0 and p6 == 1)
                        + (p6 == 0 and p7 == 1)
                        + (p7 == 0 and p8 == 1)
                        + (p8 == 0 and p9 == 1)
                        + (p9 == 0 and p2 == 1)
                    )
                    if transitions != 1:
                        continue
                    if step == 0:
                        if p2 * p4 * p6 != 0:
                            continue
                        if p4 * p6 * p8 != 0:
                            continue
                    else:
                        if p2 * p4 * p8 != 0:
                            continue
                        if p2 * p6 * p8 != 0:
                            continue
                    to_remove.append((y, x))
            if to_remove:
                changed = True
                for y, x in to_remove:
                    image[y, x] = 0
    return image.astype(bool)
