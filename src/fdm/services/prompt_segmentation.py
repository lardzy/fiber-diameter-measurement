from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2

from fdm.geometry import Point, clean_ring, distance, point_in_polygon, ring_signed_area
from fdm.runtime_logging import append_runtime_log
from fdm.settings import MagicSegmentModelVariant, bundle_resource_root


EDGE_SAM_ENCODER_FILENAME = "edge_sam_encoder.onnx"
EDGE_SAM_DECODER_FILENAME = "edge_sam_decoder.onnx"
EDGE_SAM_3X_ENCODER_FILENAME = "edge_sam_3x_encoder.onnx"
EDGE_SAM_3X_DECODER_FILENAME = "edge_sam_3x_decoder.onnx"
EDGE_SAM_TARGET_LENGTH = 1024
EDGE_SAM_PIXEL_MEAN = (123.675, 116.28, 103.53)
EDGE_SAM_PIXEL_STD = (58.395, 57.12, 57.375)


@dataclass(slots=True)
class PromptSegmentationResult:
    mask: object | None
    polygon_px: list[Point]
    area_rings_px: list[list[Point]]
    area_px: float
    metadata: dict[str, object]


@dataclass(slots=True)
class _EmbeddingEntry:
    image_embeddings: object
    original_size: tuple[int, int]


def _normalize_model_variant(model_variant: str | None) -> str:
    token = str(model_variant or "").strip()
    if token in {
        MagicSegmentModelVariant.EDGE_SAM,
        MagicSegmentModelVariant.EDGE_SAM_3X,
    }:
        return token
    return MagicSegmentModelVariant.EDGE_SAM


def edge_sam_runtime_root(model_variant: str = MagicSegmentModelVariant.EDGE_SAM) -> Path:
    normalized = _normalize_model_variant(model_variant)
    folder = "edge_sam_3x" if normalized == MagicSegmentModelVariant.EDGE_SAM_3X else "edge_sam"
    return bundle_resource_root() / "runtime" / "segment-anything" / folder


def edge_sam_model_paths(
    model_variant: str = MagicSegmentModelVariant.EDGE_SAM,
) -> tuple[Path, Path]:
    normalized = _normalize_model_variant(model_variant)
    runtime_root = edge_sam_runtime_root(normalized)
    if normalized == MagicSegmentModelVariant.EDGE_SAM_3X:
        return (
            runtime_root / EDGE_SAM_3X_ENCODER_FILENAME,
            runtime_root / EDGE_SAM_3X_DECODER_FILENAME,
        )
    return (
        runtime_root / EDGE_SAM_ENCODER_FILENAME,
        runtime_root / EDGE_SAM_DECODER_FILENAME,
    )


def magic_segment_model_label(model_variant: str) -> str:
    normalized = _normalize_model_variant(model_variant)
    if normalized == MagicSegmentModelVariant.EDGE_SAM_3X:
        return "EdgeSAM-3x"
    return "EdgeSAM"


def resolve_magic_segment_model_variant(
    requested_variant: str | None,
) -> tuple[str, str | None]:
    normalized = _normalize_model_variant(requested_variant)
    requested_paths = edge_sam_model_paths(normalized)
    if all(path.exists() for path in requested_paths):
        return normalized, None
    if normalized != MagicSegmentModelVariant.EDGE_SAM:
        fallback_paths = edge_sam_model_paths(MagicSegmentModelVariant.EDGE_SAM)
        if all(path.exists() for path in fallback_paths):
            return (
                MagicSegmentModelVariant.EDGE_SAM,
                "未找到 EdgeSAM-3x 模型文件，已自动回退到标准 EdgeSAM。",
            )
    return normalized, None


def magic_mask_area_px(mask) -> float:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency is required by the app
        raise RuntimeError("numpy is required for the magic segmentation tool.") from exc
    if mask is None:
        return 0.0
    return float(np.count_nonzero(mask))


def magic_mask_to_polygon(
    mask,
    *,
    positive_points: list[Point] | None = None,
    negative_points: list[Point] | None = None,
) -> list[Point]:
    _selected_mask, _rings, polygon, _stats = magic_mask_to_geometry(
        mask,
        positive_points=positive_points,
        negative_points=negative_points,
    )
    return polygon


def magic_mask_to_area_rings(
    mask,
    *,
    positive_points: list[Point] | None = None,
    negative_points: list[Point] | None = None,
) -> list[list[Point]]:
    _selected_mask, rings, _polygon, _stats = magic_mask_to_geometry(
        mask,
        positive_points=positive_points,
        negative_points=negative_points,
    )
    return rings


def magic_mask_to_geometry(
    mask,
    *,
    positive_points: list[Point] | None = None,
    negative_points: list[Point] | None = None,
) -> tuple[object | None, list[list[Point]], list[Point], dict[str, object]]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency is required by the app
        raise RuntimeError("numpy is required for the magic segmentation tool.") from exc
    normalized = normalize_magic_draft_mask(mask)
    stats: dict[str, object] = {
        "component_count": 0,
        "selected_positive_hits": 0,
        "selected_negative_hits": 0,
        "opened_holes": 0,
        "bridge_fallback": False,
        "bridge_method": "none",
    }
    if normalized is None or not np.any(normalized):
        return None, [], [], stats
    selected_mask, component_stats = _select_prompt_component(
        normalized,
        positive_points=positive_points or [],
        negative_points=negative_points or [],
    )
    stats.update(component_stats)
    if selected_mask is None or not np.any(selected_mask):
        return None, [], [], stats
    rings = _mask_to_area_rings(selected_mask)
    if not rings:
        return selected_mask, [], [], stats
    polygon, bridge_stats = _bridge_area_rings_to_polygon(rings, fallback_mask=selected_mask)
    stats.update(bridge_stats)
    return selected_mask, rings, polygon, stats


def normalize_magic_draft_mask(mask):
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency is required by the app
        raise RuntimeError("numpy is required for the magic segmentation tool.") from exc
    if mask is None:
        return None
    working = np.asarray(mask, dtype=bool)
    if not np.any(working):
        return None
    return working.copy()


def finalize_magic_subtraction_mask(primary_mask, subtract_mask) -> tuple[object | None, dict[str, object]]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency is required by the app
        raise RuntimeError("numpy is required for the magic segmentation tool.") from exc
    primary = normalize_magic_draft_mask(primary_mask)
    stats = {
        "opened_holes": 0,
        "discarded_fragments": False,
        "result_empty": False,
        "had_intersection": False,
    }
    if primary is None:
        stats["result_empty"] = True
        return None, stats
    subtract = normalize_magic_draft_mask(subtract_mask)
    if subtract is None:
        return primary.copy(), stats
    intersection = primary & subtract
    if not np.any(intersection):
        return primary.copy(), stats
    stats["had_intersection"] = True
    result = primary & ~subtract
    if not np.any(result):
        stats["result_empty"] = True
        return None, stats
    result, discarded_fragments = _keep_largest_component(result)
    stats["discarded_fragments"] = discarded_fragments
    if not np.any(result):
        stats["result_empty"] = True
        return None, stats
    return result, stats


def _select_prompt_component(mask, *, positive_points: list[Point], negative_points: list[Point]):
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency is required by the app
        raise RuntimeError("numpy is required for the magic segmentation tool.") from exc
    mask_uint8 = np.asarray(mask, dtype=np.uint8)
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    if component_count <= 1:
        return mask_uint8.astype(bool), {
            "component_count": 0,
            "selected_positive_hits": 0,
            "selected_negative_hits": 0,
        }
    prompt_centroid = _point_cloud_centroid(positive_points)
    best_label = 1
    best_key: tuple[float, float, float, float] | None = None
    best_stats = {
        "component_count": component_count - 1,
        "selected_positive_hits": 0,
        "selected_negative_hits": 0,
    }
    for label in range(1, component_count):
        component_mask = labels == label
        positive_hits = sum(1 for point in positive_points if _mask_contains_point(component_mask, point))
        negative_hits = sum(1 for point in negative_points if _mask_contains_point(component_mask, point))
        component_center = Point(float(centroids[label][0]), float(centroids[label][1]))
        center_distance = distance(component_center, prompt_centroid) if prompt_centroid is not None else 0.0
        area = float(stats[label, cv2.CC_STAT_AREA])
        score = (float(positive_hits), float(-negative_hits), -center_distance, area)
        if best_key is None or score > best_key:
            best_key = score
            best_label = label
            best_stats = {
                "component_count": component_count - 1,
                "selected_positive_hits": positive_hits,
                "selected_negative_hits": negative_hits,
            }
    return labels == best_label, best_stats


def _mask_contains_point(mask, point: Point) -> bool:
    height, width = mask.shape[:2]
    x = int(min(max(round(point.x), 0), width - 1))
    y = int(min(max(round(point.y), 0), height - 1))
    return bool(mask[y, x])


def _point_cloud_centroid(points: list[Point]) -> Point | None:
    if not points:
        return None
    return Point(
        sum(point.x for point in points) / len(points),
        sum(point.y for point in points) / len(points),
    )


def _mask_to_area_rings(mask) -> list[list[Point]]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency is required by the app
        raise RuntimeError("numpy is required for the magic segmentation tool.") from exc
    mask_uint8 = np.asarray(mask, dtype=np.uint8) * 255
    contours, hierarchy = cv2.findContours(mask_uint8.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None or not contours:
        return []
    outer_indices = [index for index, relation in enumerate(hierarchy[0]) if int(relation[3]) < 0]
    if not outer_indices:
        return []
    outer_index = max(outer_indices, key=lambda index: abs(cv2.contourArea(contours[index])))
    outer_ring = _contour_to_ring(contours[outer_index])
    if len(outer_ring) < 3:
        return []
    rings = [outer_ring]
    child_index = int(hierarchy[0][outer_index][2])
    while child_index >= 0:
        hole_ring = _contour_to_ring(contours[child_index])
        if len(hole_ring) >= 3 and abs(cv2.contourArea(contours[child_index])) >= 1.0:
            rings.append(hole_ring)
        child_index = int(hierarchy[0][child_index][0])
    return _normalize_area_rings(rings)


def _contour_to_ring(contour) -> list[Point]:
    points = [Point(float(point[0][0]), float(point[0][1])) for point in contour]
    return clean_ring(points, collinear_epsilon=1e-4)


def _normalize_area_rings(rings: list[list[Point]]) -> list[list[Point]]:
    if not rings:
        return []
    normalized: list[list[Point]] = []
    outer = clean_ring(rings[0], collinear_epsilon=1e-4)
    if len(outer) < 3:
        return []
    if ring_signed_area(outer) < 0:
        outer = list(reversed(outer))
    normalized.append(outer)
    for ring in rings[1:]:
        cleaned = clean_ring(ring, collinear_epsilon=1e-4)
        if len(cleaned) < 3:
            continue
        if ring_signed_area(cleaned) > 0:
            cleaned = list(reversed(cleaned))
        normalized.append(cleaned)
    return normalized


def _bridge_area_rings_to_polygon(rings: list[list[Point]], *, fallback_mask=None) -> tuple[list[Point], dict[str, object]]:
    stats: dict[str, object] = {
        "opened_holes": max(0, len(rings) - 1),
        "bridge_fallback": False,
        "bridge_method": "none" if len(rings) <= 1 else "vector",
    }
    if not rings or len(rings[0]) < 3:
        return [], stats
    if len(rings) == 1:
        return _simplify_polygon_outline(rings[0]), stats
    outer = [Point(point.x, point.y) for point in rings[0]]
    holes = [[Point(point.x, point.y) for point in ring] for ring in rings[1:]]
    ordered_holes = sorted(holes, key=lambda ring: min((point.x, point.y) for point in ring))
    for hole in ordered_holes:
        other_holes = [candidate for candidate in ordered_holes if candidate is not hole]
        bridged = _bridge_single_hole(outer, hole, other_holes)
        if bridged is None:
            polygon = _fallback_polygon_from_raster_corridors(rings, fallback_mask=fallback_mask)
            stats["bridge_fallback"] = True
            stats["bridge_method"] = "raster"
            return polygon, stats
        outer = bridged
    if not _validate_bridged_polygon(outer):
        polygon = _fallback_polygon_from_raster_corridors(rings, fallback_mask=fallback_mask)
        stats["bridge_fallback"] = True
        stats["bridge_method"] = "raster"
        return polygon, stats
    return _simplify_polygon_outline(outer), stats


def _bridge_single_hole(outer: list[Point], hole: list[Point], other_holes: list[list[Point]]) -> list[Point] | None:
    if len(outer) < 3 or len(hole) < 3:
        return None
    hole_index = _leftmost_ring_index(hole)
    bridge_start = hole[hole_index]
    outer_index = _choose_bridge_vertex(bridge_start, outer, other_holes)
    if outer_index is None:
        return None
    rotated_hole = hole[hole_index:] + hole[:hole_index]
    bridge_vertex = outer[outer_index]
    return (
        list(outer[: outer_index + 1])
        + [bridge_start]
        + rotated_hole[1:]
        + [bridge_start, bridge_vertex]
        + list(outer[outer_index + 1 :])
    )


def _leftmost_ring_index(ring: list[Point]) -> int:
    return min(range(len(ring)), key=lambda index: (ring[index].x, ring[index].y))


def _choose_bridge_vertex(hole_point: Point, outer: list[Point], holes: list[list[Point]]) -> int | None:
    candidate_indices = [index for index, point in enumerate(outer) if point.x <= hole_point.x + 1e-6]
    sorted_candidates = sorted(
        candidate_indices,
        key=lambda index: (hole_point.x - outer[index].x, distance(hole_point, outer[index])),
    )
    fallback_candidates = sorted(
        range(len(outer)),
        key=lambda index: distance(hole_point, outer[index]),
    )
    for candidates in (sorted_candidates, fallback_candidates):
        for index in candidates:
            if _segment_is_visible(hole_point, outer[index], outer, holes):
                return index
    return None


def _segment_is_visible(start: Point, end: Point, outer: list[Point], holes: list[list[Point]]) -> bool:
    midpoint = Point((start.x + end.x) / 2.0, (start.y + end.y) / 2.0)
    if not point_in_polygon(midpoint, outer):
        return False
    if any(point_in_polygon(midpoint, hole) for hole in holes if len(hole) >= 3):
        return False
    for ring in [outer, *holes]:
        for edge_start, edge_end in _iter_ring_edges(ring):
            if _segments_share_endpoint(start, end, edge_start, edge_end):
                continue
            if _segments_intersect(start, end, edge_start, edge_end):
                return False
    return True


def _iter_ring_edges(ring: list[Point]):
    for index in range(len(ring)):
        yield ring[index], ring[(index + 1) % len(ring)]


def _segments_share_endpoint(a1: Point, a2: Point, b1: Point, b2: Point, *, epsilon: float = 1e-6) -> bool:
    return (
        _points_equal(a1, b1, epsilon=epsilon)
        or _points_equal(a1, b2, epsilon=epsilon)
        or _points_equal(a2, b1, epsilon=epsilon)
        or _points_equal(a2, b2, epsilon=epsilon)
    )


def _points_equal(first: Point, second: Point, *, epsilon: float = 1e-6) -> bool:
    return abs(first.x - second.x) <= epsilon and abs(first.y - second.y) <= epsilon


def _segment_cross(a: Point, b: Point, c: Point) -> float:
    return ((b.x - a.x) * (c.y - a.y)) - ((b.y - a.y) * (c.x - a.x))


def _point_on_segment(start: Point, end: Point, point: Point, *, epsilon: float = 1e-6) -> bool:
    if abs(_segment_cross(start, end, point)) > epsilon:
        return False
    return (
        min(start.x, end.x) - epsilon <= point.x <= max(start.x, end.x) + epsilon
        and min(start.y, end.y) - epsilon <= point.y <= max(start.y, end.y) + epsilon
    )


def _segments_intersect(a1: Point, a2: Point, b1: Point, b2: Point, *, epsilon: float = 1e-6) -> bool:
    cross1 = _segment_cross(a1, a2, b1)
    cross2 = _segment_cross(a1, a2, b2)
    cross3 = _segment_cross(b1, b2, a1)
    cross4 = _segment_cross(b1, b2, a2)
    if (
        ((cross1 > epsilon and cross2 < -epsilon) or (cross1 < -epsilon and cross2 > epsilon))
        and ((cross3 > epsilon and cross4 < -epsilon) or (cross3 < -epsilon and cross4 > epsilon))
    ):
        return True
    return (
        (abs(cross1) <= epsilon and _point_on_segment(a1, a2, b1, epsilon=epsilon))
        or (abs(cross2) <= epsilon and _point_on_segment(a1, a2, b2, epsilon=epsilon))
        or (abs(cross3) <= epsilon and _point_on_segment(b1, b2, a1, epsilon=epsilon))
        or (abs(cross4) <= epsilon and _point_on_segment(b1, b2, a2, epsilon=epsilon))
    )


def _validate_bridged_polygon(polygon: list[Point]) -> bool:
    if len(polygon) < 3:
        return False
    segment_count = len(polygon)
    for first_index in range(segment_count):
        first_start = polygon[first_index]
        first_end = polygon[(first_index + 1) % segment_count]
        for second_index in range(first_index + 1, segment_count):
            if abs(first_index - second_index) <= 1:
                continue
            if first_index == 0 and second_index == segment_count - 1:
                continue
            second_start = polygon[second_index]
            second_end = polygon[(second_index + 1) % segment_count]
            if _same_segment(first_start, first_end, second_start, second_end):
                continue
            if _segments_share_endpoint(first_start, first_end, second_start, second_end):
                continue
            if _segments_intersect(first_start, first_end, second_start, second_end):
                return False
    return True


def _same_segment(a1: Point, a2: Point, b1: Point, b2: Point, *, epsilon: float = 1e-6) -> bool:
    return (
        (_points_equal(a1, b1, epsilon=epsilon) and _points_equal(a2, b2, epsilon=epsilon))
        or (_points_equal(a1, b2, epsilon=epsilon) and _points_equal(a2, b1, epsilon=epsilon))
    )


def _simplify_polygon_outline(points: list[Point]) -> list[Point]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency is required by the app
        raise RuntimeError("numpy is required for the magic segmentation tool.") from exc
    cleaned = clean_ring(points, collinear_epsilon=1e-4)
    if len(cleaned) < 3:
        return cleaned
    contour = np.array([[[point.x, point.y]] for point in cleaned], dtype=np.float32)
    perimeter = float(cv2.arcLength(contour, True))
    epsilon = max(0.35, perimeter * 0.00045)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    polygon = [Point(float(point[0][0]), float(point[0][1])) for point in approx]
    return clean_ring(polygon if len(polygon) >= 3 else cleaned, collinear_epsilon=1e-4)


def _fallback_polygon_from_raster_corridors(rings: list[list[Point]], *, fallback_mask):
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency is required by the app
        raise RuntimeError("numpy is required for the magic segmentation tool.") from exc
    if fallback_mask is None:
        return _simplify_polygon_outline(rings[0]) if rings else []
    corridor_mask = np.asarray(fallback_mask, dtype=bool).copy()
    outer = rings[0] if rings else []
    holes = rings[1:] if len(rings) > 1 else []
    for index, hole in enumerate(holes):
        if len(hole) < 3 or len(outer) < 3:
            continue
        hole_index = _leftmost_ring_index(hole)
        bridge_start = hole[hole_index]
        other_holes = [candidate for offset, candidate in enumerate(holes) if offset != index]
        outer_index = _choose_bridge_vertex(bridge_start, outer, other_holes)
        if outer_index is None:
            continue
        _carve_raster_line(corridor_mask, bridge_start, outer[outer_index])
    corridor_rings = _mask_to_area_rings(corridor_mask)
    if not corridor_rings:
        return _simplify_polygon_outline(outer) if outer else []
    return _simplify_polygon_outline(corridor_rings[0])


def _carve_raster_line(mask, start: Point, end: Point) -> None:
    height, width = mask.shape[:2]
    x0 = int(min(max(round(start.x), 0), width - 1))
    y0 = int(min(max(round(start.y), 0), height - 1))
    x1 = int(min(max(round(end.x), 0), width - 1))
    y1 = int(min(max(round(end.y), 0), height - 1))
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        mask[y0, x0] = False
        if x0 == x1 and y0 == y1:
            break
        doubled_error = 2 * error
        if doubled_error >= dy:
            error += dy
            x0 += sx
        if doubled_error <= dx:
            error += dx
            y0 += sy


def _keep_largest_component(mask):
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency is required by the app
        raise RuntimeError("numpy is required for the magic segmentation tool.") from exc
    mask_uint8 = np.asarray(mask, dtype=np.uint8)
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    if component_count <= 2:
        return mask_uint8.astype(bool), False
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(areas.argmax()) + 1
    return labels == largest_label, True


class PromptSegmentationService:
    def __init__(
        self,
        *,
        encoder_path: str | Path | None = None,
        decoder_path: str | Path | None = None,
        model_variant: str = MagicSegmentModelVariant.EDGE_SAM,
        target_length: int = EDGE_SAM_TARGET_LENGTH,
        max_cache_entries: int = 2,
    ) -> None:
        normalized_variant = _normalize_model_variant(model_variant)
        default_encoder, default_decoder = edge_sam_model_paths(normalized_variant)
        self._model_variant = normalized_variant
        self._encoder_path = Path(encoder_path) if encoder_path is not None else default_encoder
        self._decoder_path = Path(decoder_path) if decoder_path is not None else default_decoder
        self._target_length = target_length
        self._max_cache_entries = max(1, int(max_cache_entries))
        self._encoder_session = None
        self._decoder_session = None
        self._encoder_input_name = ""
        self._encoder_input_type = "tensor(uint8)"
        self._encoder_input_shape: tuple[object, ...] = ()
        self._decoder_input_names: dict[str, str] = {}
        self._embedding_cache: OrderedDict[str, _EmbeddingEntry] = OrderedDict()

    @staticmethod
    def models_ready(model_variant: str = MagicSegmentModelVariant.EDGE_SAM) -> bool:
        encoder_path, decoder_path = edge_sam_model_paths(model_variant)
        return encoder_path.exists() and decoder_path.exists()

    def clear_cache(self) -> None:
        self._embedding_cache.clear()

    def predict_polygon(
        self,
        *,
        image,
        cache_key: str,
        positive_points: list[Point],
        negative_points: list[Point],
    ) -> PromptSegmentationResult:
        if not positive_points:
            return PromptSegmentationResult(
                mask=None,
                polygon_px=[],
                area_rings_px=[],
                area_px=0.0,
                metadata={"reason": "missing_positive_prompt"},
            )
        cv_image = self._image_to_rgb_array(image)
        embedding = self._embedding_for_rgb_array(cv_image, cache_key=cache_key)
        mask = self._predict_mask_from_embedding(
            embedding,
            positive_points=positive_points,
            negative_points=negative_points,
        )
        selected_mask, area_rings, polygon, geometry_stats = magic_mask_to_geometry(
            mask,
            positive_points=positive_points,
            negative_points=negative_points,
        )
        return PromptSegmentationResult(
            mask=selected_mask.copy() if selected_mask is not None else None,
            polygon_px=polygon,
            area_rings_px=area_rings,
            area_px=magic_mask_area_px(selected_mask),
            metadata={
                "positive_points": len(positive_points),
                "negative_points": len(negative_points),
                "cache_size": len(self._embedding_cache),
                "cache_key": cache_key,
                "model_variant": self._model_variant,
                **geometry_stats,
            },
        )

    def _ensure_sessions(self) -> None:
        if self._encoder_session is not None and self._decoder_session is not None:
            return
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required for the magic segmentation tool.") from exc

        if not self._encoder_path.exists() or not self._decoder_path.exists():
            raise FileNotFoundError(
                f"未找到 {magic_segment_model_label(self._model_variant)} 模型文件，请确认对应 runtime 目录中存在 encoder/decoder ONNX。"
            )

        self._encoder_session = ort.InferenceSession(
            self._encoder_path.as_posix(),
            providers=["CPUExecutionProvider"],
        )
        self._decoder_session = ort.InferenceSession(
            self._decoder_path.as_posix(),
            providers=["CPUExecutionProvider"],
        )
        inputs = self._encoder_session.get_inputs()
        if not inputs:
            raise RuntimeError(f"{magic_segment_model_label(self._model_variant)} encoder 未暴露输入张量。")
        self._encoder_input_name = inputs[0].name
        self._encoder_input_type = str(getattr(inputs[0], "type", "") or "tensor(uint8)")
        self._encoder_input_shape = tuple(getattr(inputs[0], "shape", ()) or ())
        self._decoder_input_names = {item.name: item.name for item in self._decoder_session.get_inputs()}

    def _image_to_rgb_array(self, image):
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("numpy is required for the magic segmentation tool.") from exc
        from PySide6.QtGui import QImage

        if image is None or not hasattr(image, "isNull") or image.isNull():
            raise RuntimeError("无法读取图片: 当前图像为空。")
        rgb = image.convertToFormat(QImage.Format.Format_RGB888)
        buffer = rgb.constBits()
        array = np.frombuffer(buffer, dtype=np.uint8, count=rgb.sizeInBytes())
        array = array.reshape((rgb.height(), rgb.bytesPerLine()))
        return array[:, : rgb.width() * 3].reshape((rgb.height(), rgb.width(), 3)).copy()

    def _embedding_for_image(self, image, *, cache_key: str) -> _EmbeddingEntry:
        key = str(cache_key)
        cached = self._embedding_cache.get(key)
        if cached is not None:
            self._embedding_cache.move_to_end(key)
            return cached
        cv_image = self._image_to_rgb_array(image)
        return self._embedding_for_rgb_array(cv_image, cache_key=cache_key)

    def _embedding_for_rgb_array(self, cv_image, *, cache_key: str) -> _EmbeddingEntry:
        key = str(cache_key)
        cached = self._embedding_cache.get(key)
        if cached is not None:
            self._embedding_cache.move_to_end(key)
            return cached
        started_at = perf_counter()
        image_embeddings, original_size = self._run_encoder(cv_image)
        cached = _EmbeddingEntry(image_embeddings=image_embeddings, original_size=original_size)
        self._embedding_cache[key] = cached
        self._embedding_cache.move_to_end(key)
        while len(self._embedding_cache) > self._max_cache_entries:
            self._embedding_cache.popitem(last=False)
        elapsed_ms = (perf_counter() - started_at) * 1000.0
        if elapsed_ms >= 80.0:
            append_runtime_log(
                "Magic segmentation preprocess",
                (
                    f"elapsed_ms={elapsed_ms:.2f}, "
                    f"cache_size={len(self._embedding_cache)}, "
                    f"image_size={original_size[1]}x{original_size[0]}, "
                    f"model_variant={self._model_variant}"
                ),
            )
        return cached

    def _run_encoder(self, cv_image):
        self._ensure_sessions()
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("numpy is required for the magic segmentation tool.") from exc
        original_size = tuple(int(value) for value in cv_image.shape[:2])
        target_length = self._effective_target_length()
        target_h, target_w = self._get_preprocess_shape(original_size[0], original_size[1], target_length)
        resized = cv2.resize(cv_image, (target_w, target_h))
        transformed = resized.transpose((2, 0, 1))
        if self._requires_external_resize_norm_pad():
            transformed = self._normalize_and_pad_encoder_input(
                transformed,
                np,
                target_size=target_length,
                input_size=(target_h, target_w),
            )
        else:
            transformed = self._cast_encoder_input(transformed, np)
        transformed = transformed[None, ...]
        image_embeddings = self._encoder_session.run(
            None,
            {self._encoder_input_name: transformed},
        )[0]
        return image_embeddings, original_size

    def _effective_target_length(self) -> int:
        fixed_square_size = self._fixed_square_encoder_size()
        if fixed_square_size is not None:
            return fixed_square_size
        return self._target_length

    def _fixed_square_encoder_size(self) -> int | None:
        if len(self._encoder_input_shape) < 4:
            return None
        height = self._encoder_input_shape[2]
        width = self._encoder_input_shape[3]
        if isinstance(height, int) and isinstance(width, int) and height > 0 and height == width:
            return int(height)
        return None

    def _requires_external_resize_norm_pad(self) -> bool:
        fixed_square_size = self._fixed_square_encoder_size()
        input_type = str(self._encoder_input_type or "").strip().lower()
        return fixed_square_size is not None and input_type in {
            "tensor(float)",
            "tensor(float16)",
            "tensor(float32)",
            "tensor(double)",
        }

    def _cast_encoder_input(self, transformed, np):
        input_type = str(self._encoder_input_type or "").strip().lower()
        if input_type == "tensor(uint8)":
            return np.ascontiguousarray(transformed.astype(np.uint8, copy=False))
        if input_type in {"tensor(float)", "tensor(float32)"}:
            return np.ascontiguousarray(transformed.astype(np.float32, copy=False))
        if input_type == "tensor(float16)":
            return np.ascontiguousarray(transformed.astype(np.float16, copy=False))
        if input_type == "tensor(double)":
            return np.ascontiguousarray(transformed.astype(np.float64, copy=False))
        if input_type == "tensor(int64)":
            return np.ascontiguousarray(transformed.astype(np.int64, copy=False))
        raise RuntimeError(
            f"{magic_segment_model_label(self._model_variant)} encoder 输入类型暂不支持: {self._encoder_input_type}"
        )

    def _normalize_and_pad_encoder_input(
        self,
        transformed,
        np,
        *,
        target_size: int,
        input_size: tuple[int, int],
    ):
        normalized = transformed.astype(np.float32, copy=False)
        mean = np.asarray(EDGE_SAM_PIXEL_MEAN, dtype=np.float32).reshape(3, 1, 1)
        std = np.asarray(EDGE_SAM_PIXEL_STD, dtype=np.float32).reshape(3, 1, 1)
        normalized = (normalized - mean) / std
        padded = np.zeros((3, target_size, target_size), dtype=np.float32)
        padded[:, : input_size[0], : input_size[1]] = normalized
        return self._cast_encoder_input(padded, np)

    def _predict_mask_from_embedding(
        self,
        embedding: _EmbeddingEntry,
        *,
        positive_points: list[Point],
        negative_points: list[Point],
    ):
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("numpy is required for the magic segmentation tool.") from exc
        self._ensure_sessions()
        prompt_points = [[point.x, point.y] for point in positive_points + negative_points]
        prompt_labels = [1.0] * len(positive_points) + [0.0] * len(negative_points)
        point_coords = np.array(prompt_points, dtype=np.float32)
        point_labels = np.array(prompt_labels, dtype=np.float32)
        point_coords = self._apply_coords(point_coords, embedding.original_size).astype(np.float32)
        point_coords = np.expand_dims(point_coords, axis=0)
        point_labels = np.expand_dims(point_labels, axis=0)
        input_dict = {
            self._decoder_input_names.get("image_embeddings", "image_embeddings"): embedding.image_embeddings,
            self._decoder_input_names.get("point_coords", "point_coords"): point_coords,
            self._decoder_input_names.get("point_labels", "point_labels"): point_labels,
        }
        outputs = self._decoder_session.run(None, input_dict)
        masks = None
        for output in outputs:
            if getattr(output, "ndim", 0) >= 3:
                masks = output
                break
        if masks is None:
            raise RuntimeError(f"{magic_segment_model_label(self._model_variant)} decoder 未返回掩码张量。")
        scores = self._calculate_stability_score(masks[0], 0.0, 1.0)
        max_score_index = int(scores.argmax())
        mask = masks[0, max_score_index]
        input_size = self._get_preprocess_shape(*embedding.original_size, self._effective_target_length())
        return self._postprocess_masks(mask, input_size=input_size, original_size=embedding.original_size) > 0.0

    def _mask_to_polygon(self, mask) -> list[Point]:
        return magic_mask_to_polygon(mask)

    @staticmethod
    def _points_to_contour(points: list[Point]):
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("numpy is required for the magic segmentation tool.") from exc
        if not points:
            return np.zeros((0, 1, 2), dtype=np.float32)
        return np.array([[[point.x, point.y]] for point in points], dtype=np.float32)

    @staticmethod
    def _get_preprocess_shape(old_h: int, old_w: int, long_side_length: int) -> tuple[int, int]:
        scale = long_side_length * 1.0 / max(old_h, old_w)
        new_h, new_w = old_h * scale, old_w * scale
        return int(new_h + 0.5), int(new_w + 0.5)

    def _apply_coords(self, coords, original_size: tuple[int, int]):
        old_h, old_w = original_size
        new_h, new_w = self._get_preprocess_shape(old_h, old_w, self._effective_target_length())
        adjusted = coords.copy().astype(float)
        adjusted[..., 0] = adjusted[..., 0] * (new_w / old_w)
        adjusted[..., 1] = adjusted[..., 1] * (new_h / old_h)
        return adjusted

    @staticmethod
    def _calculate_stability_score(masks, mask_threshold: float, threshold_offset: float):
        high_threshold_mask = masks > (mask_threshold + threshold_offset)
        low_threshold_mask = masks > (mask_threshold - threshold_offset)
        intersections = high_threshold_mask & low_threshold_mask
        unions = high_threshold_mask | low_threshold_mask
        return intersections.sum(axis=(-1, -2), dtype="int32") / unions.sum(axis=(-1, -2), dtype="int32")

    def _postprocess_masks(
        self,
        mask,
        *,
        input_size: tuple[int, int],
        original_size: tuple[int, int],
    ):
        img_size = self._effective_target_length()
        resized = cv2.resize(mask, (img_size, img_size))
        cropped = resized[..., : input_size[0], : input_size[1]]
        return cv2.resize(cropped, original_size[::-1])
