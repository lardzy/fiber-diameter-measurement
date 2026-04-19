from __future__ import annotations

import cv2
import numpy as np

from fdm.geometry import Point, area_rings_bounds, clean_ring, polygon_bounds, ring_signed_area
from fdm.models import Measurement


DISPLAY_EPSILON_FACTOR = 0.0035
DISPLAY_MIN_EPSILON = 1.0


def is_magic_segment_area(measurement: Measurement) -> bool:
    return measurement.measurement_kind == "area" and measurement.mode == "magic_segment"


def invalidate_measurement_display_geometry(measurement: Measurement) -> None:
    measurement.display_polygon_px = []
    measurement.display_area_rings_px = []
    measurement.display_bounds_px = None


def ensure_measurement_display_geometry(measurement: Measurement) -> None:
    if not is_magic_segment_area(measurement):
        invalidate_measurement_display_geometry(measurement)
        return
    if measurement.display_bounds_px is not None:
        return
    rings_source = measurement.area_rings_px or ([measurement.polygon_px] if len(measurement.polygon_px) >= 3 else [])
    outline_source = measurement.polygon_px if len(measurement.polygon_px) >= 3 else (rings_source[0] if rings_source else [])
    display_rings = _simplify_area_rings(rings_source)
    display_polygon = _simplify_ring(outline_source)
    if len(display_polygon) < 3 and display_rings:
        display_polygon = list(display_rings[0])
    if not display_rings and len(display_polygon) >= 3:
        display_rings = [list(display_polygon)]
    measurement.display_polygon_px = display_polygon
    measurement.display_area_rings_px = display_rings
    if display_rings:
        measurement.display_bounds_px = area_rings_bounds(display_rings)
    elif len(display_polygon) >= 3:
        measurement.display_bounds_px = polygon_bounds(display_polygon)
    else:
        measurement.display_bounds_px = None


def area_geometry_for_display(
    measurement: Measurement,
    *,
    selected: bool,
) -> tuple[list[Point], list[list[Point]], tuple[float, float, float, float] | None]:
    if selected or not is_magic_segment_area(measurement):
        outline_points = measurement.polygon_px
        fill_rings = measurement.area_rings_px or ([measurement.polygon_px] if len(measurement.polygon_px) >= 3 else [])
        if len(outline_points) < 3 and fill_rings:
            outline_points = fill_rings[0]
        bounds = (
            area_rings_bounds(fill_rings)
            if fill_rings
            else polygon_bounds(outline_points)
            if len(outline_points) >= 3
            else None
        )
        return outline_points, fill_rings, bounds
    ensure_measurement_display_geometry(measurement)
    outline_points = measurement.display_polygon_px
    fill_rings = measurement.display_area_rings_px
    if len(outline_points) < 3 and fill_rings:
        outline_points = fill_rings[0]
    return outline_points, fill_rings, measurement.display_bounds_px


def _simplify_area_rings(rings: list[list[Point]]) -> list[list[Point]]:
    if not rings:
        return []
    simplified: list[list[Point]] = []
    for index, ring in enumerate(rings):
        reduced = _simplify_ring(ring)
        if len(reduced) < 3:
            continue
        if index == 0:
            if ring_signed_area(reduced) < 0:
                reduced = list(reversed(reduced))
        else:
            if ring_signed_area(reduced) > 0:
                reduced = list(reversed(reduced))
        simplified.append(reduced)
    return simplified


def _simplify_ring(points: list[Point]) -> list[Point]:
    cleaned = clean_ring(points, collinear_epsilon=1e-4)
    if len(cleaned) < 3:
        return cleaned
    contour = np.array([[[point.x, point.y]] for point in cleaned], dtype=np.float32)
    perimeter = float(cv2.arcLength(contour, True))
    epsilon = max(DISPLAY_MIN_EPSILON, perimeter * DISPLAY_EPSILON_FACTOR)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    simplified = [Point(float(point[0][0]), float(point[0][1])) for point in approx]
    return clean_ring(simplified if len(simplified) >= 3 else cleaned, collinear_epsilon=1e-4)
