from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

import cv2

from fdm.geometry import Point, clean_ring, distance, point_in_polygon, ring_signed_area
from fdm.runtime_logging import append_runtime_log
from fdm.settings import (
    ComplexMagicSegmentModelVariant,
    MagicSegmentToolMode,
    MagicSegmentModelVariant,
    bundle_resource_root,
)


EDGE_SAM_ENCODER_FILENAME = "edge_sam_encoder.onnx"
EDGE_SAM_DECODER_FILENAME = "edge_sam_decoder.onnx"
EDGE_SAM_3X_ENCODER_FILENAME = "edge_sam_3x_encoder.onnx"
EDGE_SAM_3X_DECODER_FILENAME = "edge_sam_3x_decoder.onnx"
LIGHT_HQ_SAM_CHECKPOINT_FILENAME = "sam_hq_vit_tiny.pth"
EFFICIENTSAM_S_CHECKPOINT_FILENAME = "efficient_sam_vits.pt"
EDGE_SAM_TARGET_LENGTH = 1024
EDGE_SAM_PIXEL_MEAN = (123.675, 116.28, 103.53)
EDGE_SAM_PIXEL_STD = (58.395, 57.12, 57.375)
ROI_CROP_MIN_SIDE = 256
ROI_CROP_MAX_INITIAL_SIDE = 768
ROI_CROP_SCALE_FACTOR = 1.8
ROI_CROP_STANDARD_MAX_ROUNDS = 4
ROI_CROP_FIBER_QUICK_MAX_ROUNDS = 3
ROI_CROP_ACCEPT_AREA_RATIO = 0.70


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


@dataclass(slots=True)
class _TorchImageEntry:
    image_embeddings: object
    original_size: tuple[int, int]
    input_size: tuple[int, int]
    interm_features: object | None = None


def _normalize_backend_id(model_variant: str | None) -> str:
    token = str(model_variant or "").strip()
    if token in {
        MagicSegmentModelVariant.EDGE_SAM,
        MagicSegmentModelVariant.EDGE_SAM_3X,
        ComplexMagicSegmentModelVariant.LIGHT_HQ_SAM,
        ComplexMagicSegmentModelVariant.EFFICIENTSAM_S,
    }:
        return token
    return MagicSegmentModelVariant.EDGE_SAM


def interactive_segmentation_runtime_root(model_variant: str = MagicSegmentModelVariant.EDGE_SAM) -> Path:
    normalized = _normalize_backend_id(model_variant)
    folder = {
        MagicSegmentModelVariant.EDGE_SAM: "edge_sam",
        MagicSegmentModelVariant.EDGE_SAM_3X: "edge_sam_3x",
        ComplexMagicSegmentModelVariant.LIGHT_HQ_SAM: "light_hq_sam",
        ComplexMagicSegmentModelVariant.EFFICIENTSAM_S: "efficient_sam_s",
    }.get(normalized, "edge_sam")
    return bundle_resource_root() / "runtime" / "segment-anything" / folder


def interactive_segmentation_model_paths(model_variant: str = MagicSegmentModelVariant.EDGE_SAM) -> tuple[Path, ...]:
    normalized = _normalize_backend_id(model_variant)
    runtime_root = interactive_segmentation_runtime_root(normalized)
    if normalized == MagicSegmentModelVariant.EDGE_SAM_3X:
        return (
            runtime_root / EDGE_SAM_3X_ENCODER_FILENAME,
            runtime_root / EDGE_SAM_3X_DECODER_FILENAME,
        )
    if normalized == MagicSegmentModelVariant.EDGE_SAM:
        return (
            runtime_root / EDGE_SAM_ENCODER_FILENAME,
            runtime_root / EDGE_SAM_DECODER_FILENAME,
        )
    if normalized == ComplexMagicSegmentModelVariant.LIGHT_HQ_SAM:
        return (runtime_root / LIGHT_HQ_SAM_CHECKPOINT_FILENAME,)
    return (runtime_root / EFFICIENTSAM_S_CHECKPOINT_FILENAME,)


def interactive_segmentation_model_label(model_variant: str) -> str:
    normalized = _normalize_backend_id(model_variant)
    if normalized == MagicSegmentModelVariant.EDGE_SAM_3X:
        return "EdgeSAM-3x"
    if normalized == ComplexMagicSegmentModelVariant.LIGHT_HQ_SAM:
        return "Light HQ-SAM"
    if normalized == ComplexMagicSegmentModelVariant.EFFICIENTSAM_S:
        return "EfficientSAM-S"
    return "EdgeSAM"


def interactive_segmentation_models_ready(model_variant: str = MagicSegmentModelVariant.EDGE_SAM) -> bool:
    return all(path.exists() for path in interactive_segmentation_model_paths(model_variant))


def resolve_interactive_segmentation_backend(
    requested_variant: str | None,
) -> tuple[str, str | None]:
    normalized = _normalize_backend_id(requested_variant)
    requested_paths = interactive_segmentation_model_paths(normalized)
    if all(path.exists() for path in requested_paths):
        return normalized, None
    if normalized != MagicSegmentModelVariant.EDGE_SAM_3X:
        return normalized, None
    fallback_paths = interactive_segmentation_model_paths(MagicSegmentModelVariant.EDGE_SAM)
    if all(path.exists() for path in fallback_paths):
        return (
            MagicSegmentModelVariant.EDGE_SAM,
            "未找到 EdgeSAM-3x 模型文件，已自动回退到标准 EdgeSAM。",
        )
    return normalized, None


def _normalize_model_variant(model_variant: str | None) -> str:
    token = _normalize_backend_id(model_variant)
    if token in {
        MagicSegmentModelVariant.EDGE_SAM,
        MagicSegmentModelVariant.EDGE_SAM_3X,
    }:
        return token
    return MagicSegmentModelVariant.EDGE_SAM


def edge_sam_runtime_root(model_variant: str = MagicSegmentModelVariant.EDGE_SAM) -> Path:
    normalized = _normalize_model_variant(model_variant)
    return interactive_segmentation_runtime_root(normalized)


def edge_sam_model_paths(
    model_variant: str = MagicSegmentModelVariant.EDGE_SAM,
) -> tuple[Path, Path]:
    normalized = _normalize_model_variant(model_variant)
    paths = interactive_segmentation_model_paths(normalized)
    return paths[0], paths[1]


def magic_segment_model_label(model_variant: str) -> str:
    return interactive_segmentation_model_label(model_variant)


def resolve_magic_segment_model_variant(
    requested_variant: str | None,
) -> tuple[str, str | None]:
    normalized, message = resolve_interactive_segmentation_backend(requested_variant)
    if normalized not in {
        MagicSegmentModelVariant.EDGE_SAM,
        MagicSegmentModelVariant.EDGE_SAM_3X,
    }:
        return MagicSegmentModelVariant.EDGE_SAM, None
    return normalized, message


def magic_mask_area_px(mask) -> float:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency is required by the app
        raise RuntimeError("numpy is required for the magic segmentation tool.") from exc
    if mask is None:
        return 0.0
    return float(np.count_nonzero(mask))


def _tool_mode_uses_auto_roi(tool_mode: str | None, *, roi_enabled: bool = False) -> bool:
    token = str(tool_mode or "").strip()
    if not roi_enabled:
        return False
    return token in {
        MagicSegmentToolMode.STANDARD,
        MagicSegmentToolMode.FIBER_QUICK,
    }


def _prompt_centroid(points: list[Point]) -> Point | None:
    if not points:
        return None
    total_x = sum(point.x for point in points)
    total_y = sum(point.y for point in points)
    return Point(total_x / len(points), total_y / len(points))


def _prompt_bbox_long_side(positive_points: list[Point], negative_points: list[Point]) -> float:
    points = list(positive_points) + list(negative_points)
    if not points:
        return 1.0
    xs = [point.x for point in points]
    ys = [point.y for point in points]
    return max(1.0, max(max(xs) - min(xs), max(ys) - min(ys)))


def _centered_square_crop(
    *,
    center: Point,
    side: int,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    image_h, image_w = image_size
    crop_side = max(1, min(int(side), image_h, image_w))
    half = crop_side / 2.0
    x0 = int(round(center.x - half))
    y0 = int(round(center.y - half))
    x0 = max(0, min(x0, image_w - crop_side))
    y0 = max(0, min(y0, image_h - crop_side))
    return x0, y0, x0 + crop_side, y0 + crop_side


def initial_interactive_segmentation_crop_box(
    *,
    image_size: tuple[int, int],
    positive_points: list[Point],
    negative_points: list[Point],
    tool_mode: str,
    roi_enabled: bool = False,
) -> tuple[int, int, int, int] | None:
    if not _tool_mode_uses_auto_roi(tool_mode, roi_enabled=roi_enabled) or not positive_points:
        return None
    image_h, image_w = image_size
    latest_positive = positive_points[-1] if positive_points else _prompt_centroid(positive_points)
    center = latest_positive or _prompt_centroid(positive_points) or Point(image_w / 2.0, image_h / 2.0)
    prompt_long_side = _prompt_bbox_long_side(positive_points, negative_points)
    initial_side = int(round(max(ROI_CROP_MIN_SIDE, 4.0 * prompt_long_side)))
    initial_side = max(ROI_CROP_MIN_SIDE, min(initial_side, ROI_CROP_MAX_INITIAL_SIDE))
    max_side = min(1024, int(round(max(image_h, image_w) * 0.6)))
    crop_side = max(ROI_CROP_MIN_SIDE, min(initial_side, max_side, image_h, image_w))
    return _centered_square_crop(center=center, side=crop_side, image_size=image_size)


def _crop_points_to_local(points: list[Point], crop_box: tuple[int, int, int, int]) -> list[Point]:
    x0, y0, _x1, _y1 = crop_box
    return [Point(point.x - x0, point.y - y0) for point in points]


def _expand_mask_from_crop(crop_mask, *, crop_box: tuple[int, int, int, int], image_shape: tuple[int, int]):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required for the magic segmentation tool.") from exc
    full_mask = np.zeros(image_shape, dtype=bool)
    if crop_mask is None:
        return full_mask
    x0, y0, x1, y1 = crop_box
    full_mask[y0:y1, x0:x1] = crop_mask.astype(bool, copy=False)
    return full_mask


def _component_bounds(mask) -> tuple[int, int, int, int] | None:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required for the magic segmentation tool.") from exc
    if mask is None or not np.any(mask):
        return None
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _roi_result_needs_expansion(
    selected_mask,
    *,
    crop_box: tuple[int, int, int, int],
) -> bool:
    bounds = _component_bounds(selected_mask)
    if bounds is None:
        return True
    x0, y0, x1, y1 = crop_box
    min_x, min_y, max_x, max_y = bounds
    touch_count = int(min_x <= x0) + int(min_y <= y0) + int(max_x >= (x1 - 1)) + int(max_y >= (y1 - 1))
    if touch_count >= 2:
        return True
    crop_area = max(1, (x1 - x0) * (y1 - y0))
    return (magic_mask_area_px(selected_mask) / crop_area) >= ROI_CROP_ACCEPT_AREA_RATIO


def qimage_to_rgb_array(image):
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
    select_prompt_component: bool = True,
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
        "hole_count": 0,
        "opened_holes": 0,
        "bridge_fallback": False,
        "bridge_method": "none",
    }
    if normalized is None or not np.any(normalized):
        return None, [], [], stats
    if select_prompt_component:
        selected_mask, component_stats = _select_prompt_component(
            normalized,
            positive_points=positive_points or [],
            negative_points=negative_points or [],
        )
        stats.update(component_stats)
    else:
        selected_mask = normalized
        stats["component_count"] = _mask_component_count(normalized)
    if selected_mask is None or not np.any(selected_mask):
        return None, [], [], stats
    rings = _mask_to_area_rings(selected_mask)
    if not rings:
        return selected_mask, [], [], stats
    stats["hole_count"] = max(0, len(rings) - 1)
    polygon = _simplify_polygon_outline(rings[0])
    if len(polygon) < 3:
        polygon = [Point(point.x, point.y) for point in rings[0]]
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


def fill_magic_draft_internal_holes(mask):
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency is required by the app
        raise RuntimeError("numpy is required for the magic segmentation tool.") from exc
    normalized = normalize_magic_draft_mask(mask)
    if normalized is None:
        return None
    mask_uint8 = np.asarray(normalized, dtype=np.uint8) * 255
    contours, hierarchy = cv2.findContours(mask_uint8.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None or not contours:
        return normalized
    filled = mask_uint8.copy()
    for index, relation in enumerate(hierarchy[0]):
        if int(relation[3]) < 0:
            continue
        cv2.drawContours(filled, contours, index, 255, thickness=cv2.FILLED)
    return filled.astype(bool)


def _combined_subtraction_mask(subtract_mask):
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency is required by the app
        raise RuntimeError("numpy is required for the magic segmentation tool.") from exc
    mask_items = subtract_mask if isinstance(subtract_mask, (list, tuple)) else [subtract_mask]
    combined = None
    for item in mask_items:
        normalized = normalize_magic_draft_mask(item)
        if normalized is None:
            continue
        if combined is None:
            combined = normalized.copy()
        else:
            combined = np.asarray(combined, dtype=bool) | np.asarray(normalized, dtype=bool)
    return combined


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
        "subtract_shape_count": 0,
    }
    if primary is None:
        stats["result_empty"] = True
        return None, stats
    subtract = _combined_subtraction_mask(subtract_mask)
    if subtract is None:
        return primary.copy(), stats
    if isinstance(subtract_mask, (list, tuple)):
        stats["subtract_shape_count"] = len([item for item in subtract_mask if normalize_magic_draft_mask(item) is not None])
    else:
        stats["subtract_shape_count"] = 1
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


def _mask_component_count(mask) -> int:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency is required by the app
        raise RuntimeError("numpy is required for the magic segmentation tool.") from exc
    mask_uint8 = np.asarray(mask, dtype=np.uint8)
    component_count, _labels, _stats, _centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    return max(0, int(component_count) - 1)


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
        tool_mode: str = MagicSegmentToolMode.STANDARD,
        roi_enabled: bool = False,
        cancel_check: Callable[[], bool] | None = None,
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
        if _tool_mode_uses_auto_roi(tool_mode, roi_enabled=roi_enabled):
            return self._predict_polygon_auto_roi(
                cv_image,
                cache_key=cache_key,
                positive_points=positive_points,
                negative_points=negative_points,
                tool_mode=tool_mode,
                cancel_check=cancel_check,
            )
        return self._predict_polygon_for_rgb_array(
            cv_image,
            cache_key=cache_key,
            positive_points=positive_points,
            negative_points=negative_points,
            metadata_extra={},
        )

    def _predict_polygon_auto_roi(
        self,
        cv_image,
        *,
        cache_key: str,
        positive_points: list[Point],
        negative_points: list[Point],
        tool_mode: str,
        cancel_check: Callable[[], bool] | None,
    ) -> PromptSegmentationResult:
        image_h, image_w = cv_image.shape[:2]
        image_long_side = max(image_h, image_w)
        latest_positive = positive_points[-1] if positive_points else _prompt_centroid(positive_points)
        center = latest_positive or _prompt_centroid(positive_points) or Point(image_w / 2.0, image_h / 2.0)
        prompt_long_side = _prompt_bbox_long_side(positive_points, negative_points)
        initial_side = int(round(max(ROI_CROP_MIN_SIDE, 4.0 * prompt_long_side)))
        initial_side = max(ROI_CROP_MIN_SIDE, min(initial_side, ROI_CROP_MAX_INITIAL_SIDE))
        max_rounds = ROI_CROP_STANDARD_MAX_ROUNDS if tool_mode == MagicSegmentToolMode.STANDARD else ROI_CROP_FIBER_QUICK_MAX_ROUNDS
        max_side = image_long_side if tool_mode == MagicSegmentToolMode.STANDARD else min(1024, int(round(image_long_side * 0.6)))
        last_result = PromptSegmentationResult(mask=None, polygon_px=[], area_rings_px=[], area_px=0.0, metadata={})
        used_full_image = False

        for round_idx in range(max_rounds):
            if cancel_check is not None and cancel_check():
                raise RuntimeError("请求已取消。")
            requested_side = int(round(initial_side * (ROI_CROP_SCALE_FACTOR**round_idx)))
            crop_side = max(ROI_CROP_MIN_SIDE, min(requested_side, max_side, image_h, image_w))
            if tool_mode == MagicSegmentToolMode.STANDARD and (round_idx == max_rounds - 1 or crop_side >= image_long_side):
                crop_box = (0, 0, image_w, image_h)
                used_full_image = True
            else:
                crop_box = _centered_square_crop(center=center, side=crop_side, image_size=(image_h, image_w))
            x0, y0, x1, y1 = crop_box
            crop_image = cv_image[y0:y1, x0:x1].copy()
            crop_positive = _crop_points_to_local(positive_points, crop_box)
            crop_negative = _crop_points_to_local(negative_points, crop_box)
            roi_signature = f"{x0}:{y0}:{x1}:{y1}"
            crop_result = self._predict_polygon_for_rgb_array(
                crop_image,
                cache_key=f"{cache_key}|roi={roi_signature}",
                positive_points=crop_positive,
                negative_points=crop_negative,
                metadata_extra={
                    "segmentation_roi_round": round_idx + 1,
                    "segmentation_used_full_image": used_full_image,
                    "segmentation_crop_box": crop_box,
                },
            )
            expanded_mask = _expand_mask_from_crop(crop_result.mask, crop_box=crop_box, image_shape=(image_h, image_w))
            selected_mask, area_rings, polygon, geometry_stats = magic_mask_to_geometry(
                expanded_mask,
                positive_points=positive_points,
                negative_points=negative_points,
            )
            last_result = PromptSegmentationResult(
                mask=selected_mask.copy() if selected_mask is not None else None,
                polygon_px=polygon,
                area_rings_px=area_rings,
                area_px=magic_mask_area_px(selected_mask),
                metadata={
                    **crop_result.metadata,
                    **geometry_stats,
                    "segmentation_roi_round": round_idx + 1,
                    "segmentation_used_full_image": used_full_image,
                    "segmentation_crop_box": crop_box,
                },
            )
            if selected_mask is not None and not _roi_result_needs_expansion(selected_mask, crop_box=crop_box):
                return last_result
            if used_full_image:
                break
        if last_result.mask is None or _roi_result_needs_expansion(
            last_result.mask,
            crop_box=last_result.metadata.get("segmentation_crop_box", (0, 0, image_w, image_h)),
        ):
            fallback_result = self._predict_polygon_for_rgb_array(
                cv_image,
                cache_key=cache_key,
                positive_points=positive_points,
                negative_points=negative_points,
                metadata_extra={
                    "segmentation_roi_round": int(last_result.metadata.get("segmentation_roi_round", 0) or 0),
                    "segmentation_used_full_image": True,
                    "segmentation_crop_box": (0, 0, image_w, image_h),
                    "segmentation_fallback_from_roi": True,
                },
            )
            if fallback_result.mask is not None:
                return fallback_result
            return PromptSegmentationResult(
                mask=None,
                polygon_px=[],
                area_rings_px=[],
                area_px=0.0,
                metadata={
                    "reason": "roi_unstable",
                    "segmentation_roi_round": int(last_result.metadata.get("segmentation_roi_round", 0) or 0),
                    "segmentation_used_full_image": True,
                    "segmentation_fallback_from_roi": True,
                },
            )
        return last_result

    def _predict_polygon_for_rgb_array(
        self,
        cv_image,
        *,
        cache_key: str,
        positive_points: list[Point],
        negative_points: list[Point],
        metadata_extra: dict[str, object],
    ) -> PromptSegmentationResult:
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
                **metadata_extra,
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
        return qimage_to_rgb_array(image)

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
        roi_keys = [item_key for item_key in self._embedding_cache.keys() if "|roi=" in item_key]
        while len(roi_keys) > 4:
            oldest_roi_key = roi_keys.pop(0)
            self._embedding_cache.pop(oldest_roi_key, None)
        non_roi_keys = [item_key for item_key in self._embedding_cache.keys() if "|roi=" not in item_key]
        while len(non_roi_keys) > self._max_cache_entries:
            oldest_key = non_roi_keys.pop(0)
            self._embedding_cache.pop(oldest_key, None)
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


def _clone_torch_cache_value(value):
    try:
        import torch
    except ImportError:
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, list):
        return [_clone_torch_cache_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_torch_cache_value(item) for item in value)
    return value


class LightHQSamPromptSegmentationService:
    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        max_cache_entries: int = 2,
    ) -> None:
        default_checkpoint = interactive_segmentation_model_paths(ComplexMagicSegmentModelVariant.LIGHT_HQ_SAM)[0]
        self._model_variant = ComplexMagicSegmentModelVariant.LIGHT_HQ_SAM
        self._checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else default_checkpoint
        self._max_cache_entries = max(1, int(max_cache_entries))
        self._model = None
        self._predictor = None
        self._embedding_cache: OrderedDict[str, _TorchImageEntry] = OrderedDict()

    @staticmethod
    def models_ready(model_variant: str = ComplexMagicSegmentModelVariant.LIGHT_HQ_SAM) -> bool:
        return interactive_segmentation_models_ready(model_variant)

    def clear_cache(self) -> None:
        self._embedding_cache.clear()

    def predict_polygon(
        self,
        *,
        image,
        cache_key: str,
        positive_points: list[Point],
        negative_points: list[Point],
        tool_mode: str = MagicSegmentToolMode.STANDARD,
        cancel_check: Callable[[], bool] | None = None,
    ) -> PromptSegmentationResult:
        if not positive_points:
            return PromptSegmentationResult(
                mask=None,
                polygon_px=[],
                area_rings_px=[],
                area_px=0.0,
                metadata={"reason": "missing_positive_prompt"},
            )
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("numpy is required for the magic segmentation tool.") from exc
        predictor = self._ensure_predictor()
        cache_entry = self._embedding_for_rgb_array(qimage_to_rgb_array(image), cache_key=cache_key)
        self._restore_predictor_cache_entry(predictor, cache_entry)
        prompt_points = np.array(
            [[point.x, point.y] for point in positive_points + negative_points],
            dtype=np.float32,
        )
        prompt_labels = np.array(
            [1] * len(positive_points) + [0] * len(negative_points),
            dtype=np.int32,
        )
        started_at = perf_counter()
        masks, scores, _logits = predictor.predict(
            point_coords=prompt_points,
            point_labels=prompt_labels,
            multimask_output=True,
            hq_token_only=True,
        )
        inference_ms = (perf_counter() - started_at) * 1000.0
        mask_index = int(scores.argmax()) if len(scores) else 0
        mask = np.asarray(masks[mask_index], dtype=bool)
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
                "inference_ms": inference_ms,
                **geometry_stats,
            },
        )

    def _ensure_predictor(self):
        if self._predictor is not None:
            return self._predictor
        try:
            import torch
            from segment_anything_hq import SamPredictor, sam_model_registry
        except ImportError as exc:
            raise RuntimeError("segment-anything-hq、timm、torch 和 torchvision 是复杂孔洞魔棒所必需的依赖。") from exc
        if not self._checkpoint_path.exists():
            raise FileNotFoundError(
                f"未找到 {interactive_segmentation_model_label(self._model_variant)} 模型文件，请确认 {self._checkpoint_path.as_posix()} 存在。"
            )
        model = sam_model_registry["vit_tiny"](checkpoint=None)
        state_dict = torch.load(str(self._checkpoint_path), map_location=torch.device("cpu"), weights_only=False)
        model.load_state_dict(state_dict)
        model.to(device=torch.device("cpu"))
        model.eval()
        self._model = model
        self._predictor = SamPredictor(model)
        return self._predictor

    def _embedding_for_rgb_array(self, cv_image, *, cache_key: str) -> _TorchImageEntry:
        key = str(cache_key)
        cached = self._embedding_cache.get(key)
        if cached is not None:
            self._embedding_cache.move_to_end(key)
            return cached
        predictor = self._ensure_predictor()
        started_at = perf_counter()
        predictor.set_image(cv_image)
        cached = _TorchImageEntry(
            image_embeddings=_clone_torch_cache_value(getattr(predictor, "features", None)),
            original_size=tuple(int(value) for value in getattr(predictor, "original_size", cv_image.shape[:2])),
            input_size=tuple(int(value) for value in getattr(predictor, "input_size", cv_image.shape[:2])),
            interm_features=_clone_torch_cache_value(getattr(predictor, "interm_features", None)),
        )
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
                    f"image_size={cached.original_size[1]}x{cached.original_size[0]}, "
                    f"model_variant={self._model_variant}"
                ),
            )
        return cached

    @staticmethod
    def _restore_predictor_cache_entry(predictor, entry: _TorchImageEntry) -> None:
        predictor.features = _clone_torch_cache_value(entry.image_embeddings)
        predictor.interm_features = _clone_torch_cache_value(entry.interm_features)
        predictor.original_size = tuple(entry.original_size)
        predictor.input_size = tuple(entry.input_size)
        predictor.is_image_set = True


class EfficientSamSPromptSegmentationService:
    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        max_cache_entries: int = 2,
    ) -> None:
        default_checkpoint = interactive_segmentation_model_paths(ComplexMagicSegmentModelVariant.EFFICIENTSAM_S)[0]
        self._model_variant = ComplexMagicSegmentModelVariant.EFFICIENTSAM_S
        self._checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else default_checkpoint
        self._max_cache_entries = max(1, int(max_cache_entries))
        self._model = None
        self._embedding_cache: OrderedDict[str, _TorchImageEntry] = OrderedDict()

    @staticmethod
    def models_ready(model_variant: str = ComplexMagicSegmentModelVariant.EFFICIENTSAM_S) -> bool:
        return interactive_segmentation_models_ready(model_variant)

    def clear_cache(self) -> None:
        self._embedding_cache.clear()

    def predict_polygon(
        self,
        *,
        image,
        cache_key: str,
        positive_points: list[Point],
        negative_points: list[Point],
        tool_mode: str = MagicSegmentToolMode.STANDARD,
        cancel_check: Callable[[], bool] | None = None,
    ) -> PromptSegmentationResult:
        if not positive_points:
            return PromptSegmentationResult(
                mask=None,
                polygon_px=[],
                area_rings_px=[],
                area_px=0.0,
                metadata={"reason": "missing_positive_prompt"},
            )
        try:
            import numpy as np
            import torch
        except ImportError as exc:
            raise RuntimeError("numpy 和 torch 是复杂孔洞魔棒所必需的依赖。") from exc
        model = self._ensure_model()
        cache_entry = self._embedding_for_rgb_array(qimage_to_rgb_array(image), cache_key=cache_key)
        prompt_points = torch.tensor(
            [[[[point.x, point.y] for point in positive_points + negative_points]]],
            dtype=torch.float32,
        )
        prompt_labels = torch.tensor(
            [[[1] * len(positive_points) + [0] * len(negative_points)]],
            dtype=torch.float32,
        )
        started_at = perf_counter()
        with torch.no_grad():
            predicted_logits, predicted_iou = model.predict_masks(
                cache_entry.image_embeddings,
                prompt_points,
                prompt_labels,
                multimask_output=True,
                input_h=cache_entry.original_size[0],
                input_w=cache_entry.original_size[1],
                output_h=cache_entry.original_size[0],
                output_w=cache_entry.original_size[1],
            )
        inference_ms = (perf_counter() - started_at) * 1000.0
        scores = predicted_iou[0, 0]
        mask_index = int(torch.argmax(scores).item()) if int(scores.numel()) > 0 else 0
        mask = torch.ge(predicted_logits[0, 0, mask_index], 0).cpu().numpy().astype(bool)
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
                "inference_ms": inference_ms,
                **geometry_stats,
            },
        )

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        try:
            import torch
            from fdm._vendor.efficient_sam.efficient_sam import build_efficient_sam
        except ImportError as exc:
            raise RuntimeError("torch 是 EfficientSAM-S 复杂孔洞魔棒所必需的依赖。") from exc
        if not self._checkpoint_path.exists():
            raise FileNotFoundError(
                f"未找到 {interactive_segmentation_model_label(self._model_variant)} 模型文件，请确认 {self._checkpoint_path.as_posix()} 存在。"
            )
        model = build_efficient_sam(
            encoder_patch_embed_dim=384,
            encoder_num_heads=6,
            checkpoint=str(self._checkpoint_path),
        )
        model.to(device=torch.device("cpu"))
        model.eval()
        self._model = model
        return model

    def _embedding_for_rgb_array(self, cv_image, *, cache_key: str) -> _TorchImageEntry:
        key = str(cache_key)
        cached = self._embedding_cache.get(key)
        if cached is not None:
            self._embedding_cache.move_to_end(key)
            return cached
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch 是 EfficientSAM-S 复杂孔洞魔棒所必需的依赖。") from exc
        model = self._ensure_model()
        image_tensor = torch.from_numpy(cv_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        started_at = perf_counter()
        with torch.no_grad():
            image_embeddings = model.get_image_embeddings(image_tensor)
        cached = _TorchImageEntry(
            image_embeddings=_clone_torch_cache_value(image_embeddings),
            original_size=(int(cv_image.shape[0]), int(cv_image.shape[1])),
            input_size=(int(cv_image.shape[0]), int(cv_image.shape[1])),
        )
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
                    f"image_size={cached.original_size[1]}x{cached.original_size[0]}, "
                    f"model_variant={self._model_variant}"
                ),
            )
        return cached


def create_interactive_segmentation_service(model_variant: str):
    normalized = _normalize_backend_id(model_variant)
    if normalized == ComplexMagicSegmentModelVariant.LIGHT_HQ_SAM:
        return LightHQSamPromptSegmentationService()
    if normalized == ComplexMagicSegmentModelVariant.EFFICIENTSAM_S:
        return EfficientSamSPromptSegmentationService()
    return PromptSegmentationService(model_variant=normalized)
