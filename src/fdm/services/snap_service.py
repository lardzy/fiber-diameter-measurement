from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

from fdm.geometry import Line, Point, clamp, direction, line_length, midpoint, normalize
from fdm.raster import RasterImage, RotatedROI, extract_rotated_roi
from fdm.services.model_provider import ModelProvider, NullModelProvider


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
        model_provider: ModelProvider | None = None,
        *,
        padding_px: int = 48,
        half_height_px: int = 64,
    ) -> None:
        self.model_provider = model_provider or NullModelProvider()
        self.padding_px = padding_px
        self.half_height_px = half_height_px

    def set_model_provider(self, provider: ModelProvider) -> None:
        self.model_provider = provider

    def snap_measurement(self, image: RasterImage, line: Line) -> SnapResult:
        if line_length(line) < 3.0:
            return SnapResult(
                status="line_too_short",
                original_line=line,
                snapped_line=None,
                diameter_px=None,
                confidence=0.0,
                debug_payload={"reason": "Input line is too short."},
            )

        roi = extract_rotated_roi(
            image,
            line,
            padding=self.padding_px,
            half_height=self.half_height_px,
        )
        model_result = self.model_provider.infer_roi(roi)
        if model_result is not None:
            mask = self._binarize_mask(model_result.mask)
            source = "model"
            base_confidence = model_result.confidence
        else:
            mask = self._segment_roi(roi.image)
            source = "fallback"
            base_confidence = 0.5

        component = self._select_component(mask, roi.midpoint)
        if not component:
            return SnapResult(
                status="component_not_found",
                original_line=line,
                snapped_line=None,
                diameter_px=None,
                confidence=0.0,
                debug_payload={"source": source},
            )

        axis = self._estimate_primary_axis(component, fallback=direction(line))
        normal = (-axis[1], axis[0])
        center = self._closest_component_point(component, roi.midpoint)
        start_roi = self._walk_to_boundary(component, center, (-normal[0], -normal[1]))
        end_roi = self._walk_to_boundary(component, center, normal)
        if start_roi is None or end_roi is None:
            return SnapResult(
                status="boundary_not_found",
                original_line=line,
                snapped_line=None,
                diameter_px=None,
                confidence=0.0,
                debug_payload={"source": source},
            )

        start_image = roi.map_roi_to_image(start_roi)
        end_image = roi.map_roi_to_image(end_roi)
        start_image = Point(
            x=clamp(start_image.x, 0.0, max(0.0, image.width - 1.0)),
            y=clamp(start_image.y, 0.0, max(0.0, image.height - 1.0)),
        )
        end_image = Point(
            x=clamp(end_image.x, 0.0, max(0.0, image.width - 1.0)),
            y=clamp(end_image.y, 0.0, max(0.0, image.height - 1.0)),
        )
        snapped_line = Line(start=start_image, end=end_image)
        diameter_px = line_length(snapped_line)
        component_density = min(1.0, len(component) / max(1, roi.width * roi.height / 20.0))
        confidence = max(0.05, min(0.99, (base_confidence * 0.7) + (component_density * 0.3)))
        return SnapResult(
            status="snapped",
            original_line=line,
            snapped_line=snapped_line,
            diameter_px=diameter_px,
            confidence=confidence,
            debug_payload={
                "source": source,
                "roi_size": [roi.width, roi.height],
                "component_size": len(component),
                "center_roi": center.to_dict(),
                "start_roi": start_roi.to_dict(),
                "end_roi": end_roi.to_dict(),
            },
        )

    def _binarize_mask(self, image: RasterImage) -> RasterImage:
        mask = RasterImage.blank(image.width, image.height, fill=0)
        for index, value in enumerate(image.pixels):
            mask.pixels[index] = 255 if value >= 128 else 0
        return mask

    def _segment_roi(self, image: RasterImage) -> RasterImage:
        center_y = image.height // 2
        global_mean = image.mean()
        seed_values = []
        center_x = image.width // 2
        for y in range(max(0, center_y - 3), min(image.height, center_y + 4)):
            for x in range(max(0, center_x - 3), min(image.width, center_x + 4)):
                seed_values.append(image.get(x, y))
        seed_mean = sum(seed_values) / max(1, len(seed_values))
        darker_fiber = seed_mean < global_mean
        threshold = (seed_mean + global_mean) / 2.0
        mask = RasterImage.blank(image.width, image.height, fill=0)
        for index, value in enumerate(image.pixels):
            if darker_fiber:
                mask.pixels[index] = 255 if value <= threshold else 0
            else:
                mask.pixels[index] = 255 if value >= threshold else 0
        mask = self._close(mask, iterations=1)
        mask = self._open(mask, iterations=1)
        return mask

    def _open(self, image: RasterImage, *, iterations: int) -> RasterImage:
        result = image
        for _ in range(iterations):
            result = self._dilate(self._erode(result))
        return result

    def _close(self, image: RasterImage, *, iterations: int) -> RasterImage:
        result = image
        for _ in range(iterations):
            result = self._erode(self._dilate(result))
        return result

    def _erode(self, image: RasterImage) -> RasterImage:
        output = RasterImage.blank(image.width, image.height, fill=0)
        for y in range(image.height):
            for x in range(image.width):
                keep = True
                for ny in range(y - 1, y + 2):
                    for nx in range(x - 1, x + 2):
                        if image.get(nx, ny, default=0) < 128:
                            keep = False
                            break
                    if not keep:
                        break
                output.set(x, y, 255 if keep else 0)
        return output

    def _dilate(self, image: RasterImage) -> RasterImage:
        output = RasterImage.blank(image.width, image.height, fill=0)
        for y in range(image.height):
            for x in range(image.width):
                active = False
                for ny in range(y - 1, y + 2):
                    for nx in range(x - 1, x + 2):
                        if image.get(nx, ny, default=0) >= 128:
                            active = True
                            break
                    if active:
                        break
                output.set(x, y, 255 if active else 0)
        return output

    def _select_component(self, mask: RasterImage, roi_midpoint: Point) -> set[tuple[int, int]]:
        visited: set[tuple[int, int]] = set()
        best_component: set[tuple[int, int]] = set()
        best_score = float("-inf")
        for y in range(mask.height):
            for x in range(mask.width):
                if mask.get(x, y, default=0) < 128 or (x, y) in visited:
                    continue
                component = self._collect_component(mask, x, y, visited)
                if len(component) < 8:
                    continue
                cx = sum(point[0] for point in component) / len(component)
                cy = sum(point[1] for point in component) / len(component)
                distance_score = math.hypot(cx - roi_midpoint.x, cy - roi_midpoint.y)
                score = len(component) - (distance_score * 10.0)
                if score > best_score:
                    best_score = score
                    best_component = component
        return best_component

    def _collect_component(
        self,
        mask: RasterImage,
        start_x: int,
        start_y: int,
        visited: set[tuple[int, int]],
    ) -> set[tuple[int, int]]:
        stack = [(start_x, start_y)]
        component: set[tuple[int, int]] = set()
        while stack:
            x, y = stack.pop()
            if (x, y) in visited:
                continue
            visited.add((x, y))
            if mask.get(x, y, default=0) < 128:
                continue
            component.add((x, y))
            for ny in range(y - 1, y + 2):
                for nx in range(x - 1, x + 2):
                    if (nx, ny) not in visited and mask.in_bounds(nx, ny):
                        stack.append((nx, ny))
        return component

    def _estimate_primary_axis(
        self,
        component: set[tuple[int, int]],
        *,
        fallback: tuple[float, float],
    ) -> tuple[float, float]:
        if len(component) < 2:
            return normalize(fallback)
        mean_x = sum(point[0] for point in component) / len(component)
        mean_y = sum(point[1] for point in component) / len(component)
        cov_xx = 0.0
        cov_xy = 0.0
        cov_yy = 0.0
        for x, y in component:
            dx = x - mean_x
            dy = y - mean_y
            cov_xx += dx * dx
            cov_xy += dx * dy
            cov_yy += dy * dy
        if cov_xy == 0 and cov_xx == cov_yy:
            return normalize(fallback)
        angle = 0.5 * math.atan2(2.0 * cov_xy, cov_xx - cov_yy)
        return normalize((math.cos(angle), math.sin(angle)))

    def _closest_component_point(self, component: set[tuple[int, int]], point: Point) -> Point:
        best = None
        best_distance = float("inf")
        for x, y in component:
            pixel_center = Point(x=float(x), y=float(y))
            pixel_distance = math.hypot(pixel_center.x - point.x, pixel_center.y - point.y)
            if pixel_distance < best_distance:
                best_distance = pixel_distance
                best = pixel_center
        return best or point

    def _walk_to_boundary(
        self,
        component: set[tuple[int, int]],
        start: Point,
        direction_vector: tuple[float, float],
        *,
        step: float = 0.35,
        max_steps: int = 1024,
    ) -> Point | None:
        dx, dy = normalize(direction_vector)
        current = Point(start.x, start.y)
        last_inside = None
        found_inside = False
        for _ in range(max_steps):
            sample_key = (int(round(current.x)), int(round(current.y)))
            if sample_key in component:
                last_inside = Point(current.x, current.y)
                found_inside = True
            elif found_inside:
                break
            current = Point(current.x + (dx * step), current.y + (dy * step))
        return last_inside
