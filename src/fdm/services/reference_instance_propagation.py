from __future__ import annotations

from dataclasses import dataclass
from math import exp, log

import cv2

from fdm.geometry import Point, area_rings_bounds, polygon_bounds
from fdm.services.prompt_segmentation import PromptSegmentationService, magic_mask_area_px, qimage_to_rgb_array
from fdm.settings import MagicSegmentToolMode


TEMPLATE_MATCH_SCALES = (0.75, 0.9, 1.0, 1.1, 1.25)
TEMPLATE_MATCH_THRESHOLD = 0.46
MAX_TEMPLATE_PROPOSALS = 18
MAX_REFINED_CANDIDATES = 12
REFINED_SCORE_THRESHOLD = 0.48
DEDUP_IOU_THRESHOLD = 0.55


@dataclass(slots=True)
class ReferenceCandidateProposal:
    center_px: Point
    bbox: tuple[int, int, int, int]
    confidence: float


@dataclass(slots=True)
class ReferenceInstanceCandidate:
    polygon_px: list[Point]
    area_rings_px: list[list[Point]]
    confidence: float
    bbox: tuple[int, int, int, int]


@dataclass(slots=True)
class ReferenceInstancePropagationResult:
    reference_polygon_px: list[Point]
    reference_area_rings_px: list[list[Point]]
    candidates: list[ReferenceInstanceCandidate]
    metadata: dict[str, object]


class ReferenceCandidateFinder:
    def find_candidates(
        self,
        *,
        gray_image,
        reference_mask,
        reference_bbox: tuple[int, int, int, int],
    ) -> list[ReferenceCandidateProposal]:
        raise NotImplementedError


class TemplateMatchCandidateFinder(ReferenceCandidateFinder):
    def find_candidates(
        self,
        *,
        gray_image,
        reference_mask,
        reference_bbox: tuple[int, int, int, int],
    ) -> list[ReferenceCandidateProposal]:
        import numpy as np

        x0, y0, x1, y1 = reference_bbox
        template = gray_image[y0:y1, x0:x1]
        mask_crop = reference_mask[y0:y1, x0:x1]
        if template.size == 0 or mask_crop.size == 0:
            return []
        foreground_pixels = template[mask_crop]
        background_value = int(round(float(foreground_pixels.mean()))) if int(mask_crop.sum()) > 0 else int(template.mean())
        template = template.copy()
        template[~mask_crop] = background_value
        proposals: list[ReferenceCandidateProposal] = [
            ReferenceCandidateProposal(
                center_px=Point((x0 + x1) * 0.5, (y0 + y1) * 0.5),
                bbox=(x0, y0, x1, y1),
                confidence=1.0,
            )
        ]
        image_h, image_w = gray_image.shape[:2]
        for scale in TEMPLATE_MATCH_SCALES:
            scaled_w = max(12, int(round(template.shape[1] * scale)))
            scaled_h = max(12, int(round(template.shape[0] * scale)))
            if scaled_w >= image_w or scaled_h >= image_h:
                continue
            scaled_template = cv2.resize(template, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
            response = cv2.matchTemplate(gray_image, scaled_template, cv2.TM_CCOEFF_NORMED)
            working = response.copy()
            suppression_rx = max(8, scaled_w // 3)
            suppression_ry = max(8, scaled_h // 3)
            while len(proposals) < MAX_TEMPLATE_PROPOSALS:
                _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(working)
                if float(max_val) < TEMPLATE_MATCH_THRESHOLD:
                    break
                left = int(max_loc[0])
                top = int(max_loc[1])
                bbox = (left, top, left + scaled_w, top + scaled_h)
                center = Point(left + scaled_w * 0.5, top + scaled_h * 0.5)
                proposals.append(
                    ReferenceCandidateProposal(
                        center_px=center,
                        bbox=bbox,
                        confidence=float(max_val),
                    )
                )
                x_from = max(0, left - suppression_rx)
                y_from = max(0, top - suppression_ry)
                x_to = min(working.shape[1], left + scaled_w + suppression_rx)
                y_to = min(working.shape[0], top + scaled_h + suppression_ry)
                working[y_from:y_to, x_from:x_to] = -1.0
        proposals.sort(key=lambda item: item.confidence, reverse=True)
        unique: list[ReferenceCandidateProposal] = []
        for proposal in proposals:
            if any(
                abs(existing.center_px.x - proposal.center_px.x) < 6.0
                and abs(existing.center_px.y - proposal.center_px.y) < 6.0
                for existing in unique
            ):
                continue
            unique.append(proposal)
        return unique[:MAX_TEMPLATE_PROPOSALS]


class MatcherCandidateFinder(ReferenceCandidateFinder):
    def find_candidates(
        self,
        *,
        gray_image,
        reference_mask,
        reference_bbox: tuple[int, int, int, int],
    ) -> list[ReferenceCandidateProposal]:
        raise NotImplementedError("MatcherCandidateFinder will be introduced in Phase 2.")


class ReferenceInstancePropagationService:
    def __init__(
        self,
        *,
        model_variant: str,
        candidate_finder: ReferenceCandidateFinder | None = None,
    ) -> None:
        self._prompt_service = PromptSegmentationService(model_variant=model_variant)
        self._candidate_finder = candidate_finder or TemplateMatchCandidateFinder()

    def clear_cache(self) -> None:
        self._prompt_service.clear_cache()

    def propagate_from_reference(
        self,
        *,
        image,
        cache_key: str,
        reference_box: tuple[Point, Point] | None = None,
        reference_polygon_px: list[Point] | None = None,
        reference_area_rings_px: list[list[Point]] | None = None,
    ) -> ReferenceInstancePropagationResult:
        import numpy as np

        rgb = qimage_to_rgb_array(image)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        image_size = (int(rgb.shape[1]), int(rgb.shape[0]))
        reference_mask = None
        reference_polygon = _clone_points(reference_polygon_px or [])
        reference_rings = _clone_rings(reference_area_rings_px or [])
        if reference_box is not None:
            prompt_result = self._reference_from_box(
                image=image,
                cache_key=cache_key,
                reference_box=reference_box,
            )
            reference_polygon = _clone_points(prompt_result.polygon_px)
            reference_rings = _clone_rings(prompt_result.area_rings_px)
            reference_mask = np.asarray(prompt_result.mask, dtype=bool) if prompt_result.mask is not None else None
        else:
            reference_mask = geometry_to_mask(
                image_size=image_size,
                polygon_px=reference_polygon,
                area_rings_px=reference_rings,
            )
        if reference_mask is None or not reference_polygon:
            return ReferenceInstancePropagationResult(
                reference_polygon_px=[],
                reference_area_rings_px=[],
                candidates=[],
                metadata={"reason": "missing_reference_instance"},
            )
        reference_bbox = mask_bounds(reference_mask)
        if reference_bbox is None:
            return ReferenceInstancePropagationResult(
                reference_polygon_px=[],
                reference_area_rings_px=[],
                candidates=[],
                metadata={"reason": "empty_reference_mask"},
            )
        reference_descriptor = describe_mask(gray, reference_mask, reference_bbox)
        proposals = self._candidate_finder.find_candidates(
            gray_image=gray,
            reference_mask=reference_mask,
            reference_bbox=reference_bbox,
        )
        refined: list[tuple[ReferenceInstanceCandidate, object]] = []
        for proposal in proposals:
            candidate = self._refine_candidate(
                image=image,
                cache_key=cache_key,
                proposal=proposal,
                image_size=image_size,
                gray_image=gray,
                reference_descriptor=reference_descriptor,
            )
            if candidate is None:
                continue
            candidate_item, candidate_mask = candidate
            if candidate_item.confidence < REFINED_SCORE_THRESHOLD:
                continue
            if any(
                area_geometry_iou(
                    candidate_item.polygon_px,
                    candidate_item.area_rings_px,
                    existing.polygon_px,
                    existing.area_rings_px,
                ) >= DEDUP_IOU_THRESHOLD
                for existing, _existing_mask in refined
            ):
                continue
            refined.append((candidate_item, candidate_mask))
        refined.sort(key=lambda item: item[0].confidence, reverse=True)
        candidates = [item[0] for item in refined[:MAX_REFINED_CANDIDATES]]
        return ReferenceInstancePropagationResult(
            reference_polygon_px=reference_polygon,
            reference_area_rings_px=reference_rings,
            candidates=candidates,
            metadata={
                "candidate_count": len(candidates),
                "proposal_count": len(proposals),
                "finder": type(self._candidate_finder).__name__,
                "model_variant": self._prompt_service._model_variant,  # noqa: SLF001
                "reference_area_px": magic_mask_area_px(reference_mask),
            },
        )

    def _reference_from_box(
        self,
        *,
        image,
        cache_key: str,
        reference_box: tuple[Point, Point],
    ):
        image_size = (int(image.width()), int(image.height()))
        bbox = normalize_box(reference_box[0], reference_box[1], image_size=image_size)
        positive_points, negative_points = prompt_points_for_box(bbox, image_size=image_size)
        return self._prompt_service.predict_polygon(
            image=image,
            cache_key=cache_key,
            positive_points=positive_points,
            negative_points=negative_points,
            tool_mode=MagicSegmentToolMode.REFERENCE,
        )

    def _refine_candidate(
        self,
        *,
        image,
        cache_key: str,
        proposal: ReferenceCandidateProposal,
        image_size: tuple[int, int],
        gray_image,
        reference_descriptor: dict[str, object],
    ) -> tuple[ReferenceInstanceCandidate, object] | None:
        import numpy as np

        positive_points, negative_points = prompt_points_for_box(proposal.bbox, image_size=image_size)
        result = self._prompt_service.predict_polygon(
            image=image,
            cache_key=cache_key,
            positive_points=positive_points,
            negative_points=negative_points,
            tool_mode=MagicSegmentToolMode.REFERENCE,
        )
        if result.mask is None or len(result.polygon_px) < 3:
            return None
        candidate_mask = np.asarray(result.mask, dtype=bool)
        candidate_bbox = mask_bounds(candidate_mask)
        if candidate_bbox is None:
            return None
        candidate_descriptor = describe_mask(gray_image, candidate_mask, candidate_bbox)
        similarity = descriptor_similarity(reference_descriptor, candidate_descriptor)
        final_score = (0.35 * float(proposal.confidence)) + (0.65 * similarity)
        return (
            ReferenceInstanceCandidate(
                polygon_px=_clone_points(result.polygon_px),
                area_rings_px=_clone_rings(result.area_rings_px),
                confidence=float(final_score),
                bbox=candidate_bbox,
            ),
            candidate_mask,
        )


def normalize_box(
    start: Point,
    end: Point,
    *,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    image_w, image_h = image_size
    x0 = int(round(min(start.x, end.x)))
    y0 = int(round(min(start.y, end.y)))
    x1 = int(round(max(start.x, end.x)))
    y1 = int(round(max(start.y, end.y)))
    x0 = max(0, min(x0, image_w - 1))
    y0 = max(0, min(y0, image_h - 1))
    x1 = max(x0 + 1, min(x1, image_w - 1))
    y1 = max(y0 + 1, min(y1, image_h - 1))
    return x0, y0, x1 + 1, y1 + 1


def prompt_points_for_box(
    bbox: tuple[int, int, int, int],
    *,
    image_size: tuple[int, int],
) -> tuple[list[Point], list[Point]]:
    x0, y0, x1, y1 = bbox
    image_w, image_h = image_size
    width = max(1.0, float(x1 - x0))
    height = max(1.0, float(y1 - y0))
    margin = max(6.0, 0.14 * max(width, height))
    positive = [Point((x0 + x1) * 0.5, (y0 + y1) * 0.5)]
    negatives = [
        Point(max(0.0, x0 - margin), positive[0].y),
        Point(min(image_w - 1.0, x1 + margin), positive[0].y),
        Point(positive[0].x, max(0.0, y0 - margin)),
        Point(positive[0].x, min(image_h - 1.0, y1 + margin)),
        Point(max(0.0, x0 - margin), max(0.0, y0 - margin)),
        Point(min(image_w - 1.0, x1 + margin), max(0.0, y0 - margin)),
        Point(max(0.0, x0 - margin), min(image_h - 1.0, y1 + margin)),
        Point(min(image_w - 1.0, x1 + margin), min(image_h - 1.0, y1 + margin)),
    ]
    deduped: list[Point] = []
    for point in negatives:
        if any(abs(existing.x - point.x) < 1.0 and abs(existing.y - point.y) < 1.0 for existing in deduped):
            continue
        deduped.append(point)
    return positive, deduped


def geometry_to_mask(
    *,
    image_size: tuple[int, int],
    polygon_px: list[Point],
    area_rings_px: list[list[Point]],
):
    import numpy as np

    image_w, image_h = image_size
    if image_w <= 0 or image_h <= 0:
        return None
    mask = np.zeros((image_h, image_w), dtype=np.uint8)
    if area_rings_px:
        outer = area_rings_px[0] if area_rings_px else []
        if len(outer) >= 3:
            cv2.fillPoly(mask, [_points_to_contour(outer)], 1)
        for hole in area_rings_px[1:]:
            if len(hole) >= 3:
                cv2.fillPoly(mask, [_points_to_contour(hole)], 0)
        return mask.astype(bool)
    if len(polygon_px) >= 3:
        cv2.fillPoly(mask, [_points_to_contour(polygon_px)], 1)
        return mask.astype(bool)
    return None


def mask_bounds(mask) -> tuple[int, int, int, int] | None:
    import numpy as np

    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def describe_mask(gray_image, mask, bbox: tuple[int, int, int, int]) -> dict[str, object]:
    import numpy as np

    x0, y0, x1, y1 = bbox
    area = float(np.count_nonzero(mask))
    width = max(1.0, float(x1 - x0))
    height = max(1.0, float(y1 - y0))
    aspect = width / height
    mask_uint8 = np.where(mask, 255, 0).astype(np.uint8)
    moments = cv2.moments(mask_uint8)
    hu = cv2.HuMoments(moments).flatten()
    hu = np.sign(hu) * np.log10(np.abs(hu) + 1e-9)
    pixels = gray_image[mask]
    if pixels.size == 0:
        pixels = gray_image[y0:y1, x0:x1].reshape((-1,))
    hist = cv2.calcHist([pixels.astype(np.uint8)], [0], None, [16], [0, 256])
    hist = cv2.normalize(hist, None).flatten()
    sobel_x = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(sobel_x, sobel_y)
    gradient_mean = float(grad[mask].mean()) if int(mask.sum()) > 0 else 0.0
    return {
        "area": area,
        "aspect": aspect,
        "hu": hu,
        "hist": hist,
        "gradient_mean": gradient_mean,
    }


def descriptor_similarity(reference: dict[str, object], candidate: dict[str, object]) -> float:
    import numpy as np

    ref_area = max(1.0, float(reference["area"]))
    cand_area = max(1.0, float(candidate["area"]))
    ref_aspect = max(1e-4, float(reference["aspect"]))
    cand_aspect = max(1e-4, float(candidate["aspect"]))
    area_score = exp(-abs(log(cand_area / ref_area)))
    aspect_score = exp(-abs(log(cand_aspect / ref_aspect)))
    hu_distance = float(np.mean(np.abs(reference["hu"] - candidate["hu"])))
    hu_score = 1.0 / (1.0 + hu_distance)
    hist_score = float(cv2.compareHist(reference["hist"].astype("float32"), candidate["hist"].astype("float32"), cv2.HISTCMP_CORREL))
    hist_score = max(0.0, min(1.0, (hist_score + 1.0) * 0.5))
    ref_grad = max(1e-3, float(reference["gradient_mean"]))
    cand_grad = max(1e-3, float(candidate["gradient_mean"]))
    gradient_score = exp(-abs(log(cand_grad / ref_grad)))
    return float((0.22 * area_score) + (0.18 * aspect_score) + (0.22 * hu_score) + (0.22 * hist_score) + (0.16 * gradient_score))


def area_geometry_iou(
    left_polygon_px: list[Point],
    left_area_rings_px: list[list[Point]],
    right_polygon_px: list[Point],
    right_area_rings_px: list[list[Point]],
) -> float:
    import numpy as np

    left_outline = left_area_rings_px[0] if left_area_rings_px else left_polygon_px
    right_outline = right_area_rings_px[0] if right_area_rings_px else right_polygon_px
    if len(left_outline) < 3 or len(right_outline) < 3:
        return 0.0
    left_bounds = area_rings_bounds(left_area_rings_px) if left_area_rings_px else polygon_bounds(left_polygon_px)
    right_bounds = area_rings_bounds(right_area_rings_px) if right_area_rings_px else polygon_bounds(right_polygon_px)
    min_x = int(min(left_bounds[0], right_bounds[0])) - 2
    min_y = int(min(left_bounds[1], right_bounds[1])) - 2
    max_x = int(max(left_bounds[2], right_bounds[2])) + 2
    max_y = int(max(left_bounds[3], right_bounds[3])) + 2
    width = max(1, max_x - min_x + 1)
    height = max(1, max_y - min_y + 1)
    left_mask = np.zeros((height, width), dtype=np.uint8)
    right_mask = np.zeros((height, width), dtype=np.uint8)
    _fill_local_geometry_mask(left_mask, left_polygon_px, left_area_rings_px, offset_x=min_x, offset_y=min_y)
    _fill_local_geometry_mask(right_mask, right_polygon_px, right_area_rings_px, offset_x=min_x, offset_y=min_y)
    intersection = int(np.logical_and(left_mask > 0, right_mask > 0).sum())
    if intersection <= 0:
        return 0.0
    union = int(np.logical_or(left_mask > 0, right_mask > 0).sum())
    if union <= 0:
        return 0.0
    return float(intersection / union)


def _fill_local_geometry_mask(mask, polygon_px, area_rings_px, *, offset_x: int, offset_y: int) -> None:
    if area_rings_px:
        outer = area_rings_px[0]
        if len(outer) >= 3:
            cv2.fillPoly(mask, [_points_to_contour(outer, offset_x=offset_x, offset_y=offset_y)], 1)
        for hole in area_rings_px[1:]:
            if len(hole) >= 3:
                cv2.fillPoly(mask, [_points_to_contour(hole, offset_x=offset_x, offset_y=offset_y)], 0)
        return
    if len(polygon_px) >= 3:
        cv2.fillPoly(mask, [_points_to_contour(polygon_px, offset_x=offset_x, offset_y=offset_y)], 1)


def _points_to_contour(points: list[Point], *, offset_x: int = 0, offset_y: int = 0):
    import numpy as np

    return np.array(
        [[[int(round(point.x)) - offset_x, int(round(point.y)) - offset_y]] for point in points],
        dtype=np.int32,
    )


def _clone_points(points: list[Point]) -> list[Point]:
    return [Point(float(point.x), float(point.y)) for point in points]


def _clone_rings(area_rings: list[list[Point]]) -> list[list[Point]]:
    return [_clone_points(ring) for ring in area_rings]
