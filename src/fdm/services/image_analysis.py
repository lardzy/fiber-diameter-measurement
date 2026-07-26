"""UI-independent image and measurement analysis kernels.

The request and result records in this module are immutable snapshots.  They
contain no ``QObject`` or mutable project-model references and can therefore be
used by generation-guarded background workers.

Area geometry is never simplified here.  When ``exact_area_px`` is supplied it
has priority over the vector-derived area, while perimeter, centroid, Feret
diameters and masks continue to use the original rings.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Iterable, Sequence, TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

from fdm.geometry import Point, area_rings_area_and_centroid


Coordinate: TypeAlias = tuple[float, float]
ImmutableRing: TypeAlias = tuple[Coordinate, ...]
ImmutableRings: TypeAlias = tuple[ImmutableRing, ...]


@dataclass(frozen=True, slots=True)
class ShapeAnalysisRequest:
    rings: ImmutableRings
    exact_area_px: float | None = None
    pixel_size_x: float = 1.0
    pixel_size_y: float = 1.0
    unit: str = "px"
    request_id: str = ""
    generation: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "rings", _freeze_rings(self.rings))
        exact = self.exact_area_px
        if exact is not None and (not math.isfinite(float(exact)) or float(exact) < 0):
            raise ValueError("exact_area_px must be a finite non-negative number.")
        _require_positive("pixel_size_x", self.pixel_size_x)
        _require_positive("pixel_size_y", self.pixel_size_y)
        object.__setattr__(self, "exact_area_px", None if exact is None else float(exact))
        object.__setattr__(self, "pixel_size_x", float(self.pixel_size_x))
        object.__setattr__(self, "pixel_size_y", float(self.pixel_size_y))
        object.__setattr__(self, "unit", str(self.unit or "px"))
        object.__setattr__(self, "request_id", str(self.request_id))
        object.__setattr__(self, "generation", int(self.generation))

    @classmethod
    def create(
        cls,
        rings: Iterable[Iterable[Any]],
        **kwargs: Any,
    ) -> "ShapeAnalysisRequest":
        return cls(rings=_freeze_rings(rings), **kwargs)


@dataclass(frozen=True, slots=True)
class ShapeAnalysisResult:
    area_px: float
    vector_area_px: float
    area: float
    centroid_px: Coordinate
    centroid: Coordinate
    outer_perimeter_px: float
    hole_perimeter_px: float
    total_perimeter_px: float
    outer_perimeter: float
    hole_perimeter: float
    total_perimeter: float
    bounds_px: tuple[float, float, float, float]
    hole_count: int
    hole_area_px: float
    equivalent_circle_diameter: float | None
    feret_max: float | None
    feret_min: float | None
    feret_angle_degrees: float | None
    ellipse_major: float | None
    ellipse_minor: float | None
    ellipse_angle_degrees: float | None
    circularity: float | None
    aspect_ratio: float | None
    roundness: float | None
    solidity: float | None
    unit: str
    area_from_exact_mask: bool
    warnings: tuple[str, ...] = ()
    request_id: str = ""
    generation: int = 0


@dataclass(frozen=True, slots=True)
class IntensityAnalysisRequest:
    image: NDArray[Any]
    roi_mask: NDArray[np.bool_] | None = None
    rings: ImmutableRings = ()
    channel: str = "luminance"
    percentile_levels: tuple[float, ...] = (10.0, 25.0, 50.0, 75.0, 90.0)
    request_id: str = ""
    generation: int = 0

    def __post_init__(self) -> None:
        image = _freeze_image(self.image)
        object.__setattr__(self, "image", image)
        object.__setattr__(
            self,
            "roi_mask",
            None
            if self.roi_mask is None
            else _freeze_mask(self.roi_mask, image.shape[:2]),
        )
        object.__setattr__(self, "rings", _freeze_rings(self.rings))
        levels = tuple(float(value) for value in self.percentile_levels)
        if any(not math.isfinite(value) or value < 0 or value > 100 for value in levels):
            raise ValueError("percentile levels must be finite values between 0 and 100.")
        object.__setattr__(self, "percentile_levels", levels)
        object.__setattr__(self, "channel", str(self.channel))
        object.__setattr__(self, "request_id", str(self.request_id))
        object.__setattr__(self, "generation", int(self.generation))


@dataclass(frozen=True, slots=True)
class IntensityAnalysisResult:
    included_pixel_count: int
    valid_pixel_count: int
    non_finite_count: int
    mean: float | None
    median: float | None
    stddev: float | None
    minimum: float | None
    maximum: float | None
    integrated_density: float | None
    intensity_centroid_px: Coordinate | None
    percentiles: tuple[tuple[float, float], ...]
    channel: str
    request_id: str = ""
    generation: int = 0


@dataclass(frozen=True, slots=True)
class HistogramRequest:
    image: NDArray[Any]
    roi_mask: NDArray[np.bool_] | None = None
    rings: ImmutableRings = ()
    channel: str = "luminance"
    bins: int = 256
    value_range: tuple[float, float] | None = None
    request_id: str = ""
    generation: int = 0

    def __post_init__(self) -> None:
        image = _freeze_image(self.image)
        object.__setattr__(self, "image", image)
        object.__setattr__(
            self,
            "roi_mask",
            None
            if self.roi_mask is None
            else _freeze_mask(self.roi_mask, image.shape[:2]),
        )
        object.__setattr__(self, "rings", _freeze_rings(self.rings))
        if int(self.bins) < 1 or int(self.bins) > 65536:
            raise ValueError("histogram bins must be between 1 and 65536.")
        object.__setattr__(self, "bins", int(self.bins))
        if self.value_range is not None:
            low, high = (float(self.value_range[0]), float(self.value_range[1]))
            if not math.isfinite(low) or not math.isfinite(high) or high <= low:
                raise ValueError("histogram range must be finite and increasing.")
            object.__setattr__(self, "value_range", (low, high))
        object.__setattr__(self, "channel", str(self.channel))
        object.__setattr__(self, "request_id", str(self.request_id))
        object.__setattr__(self, "generation", int(self.generation))


@dataclass(frozen=True, slots=True)
class HistogramResult:
    counts: tuple[int, ...]
    edges: tuple[float, ...]
    included_pixel_count: int
    non_finite_count: int
    channel: str
    request_id: str = ""
    generation: int = 0


@dataclass(frozen=True, slots=True)
class IntensityProfileRequest:
    image: NDArray[Any]
    points: tuple[Coordinate, ...]
    line_width: float = 1.0
    sample_spacing: float = 1.0
    pixel_size_x: float = 1.0
    pixel_size_y: float = 1.0
    channel: str = "luminance"
    request_id: str = ""
    generation: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "image", _freeze_image(self.image))
        points = _freeze_ring(self.points)
        if len(points) < 2:
            raise ValueError("an intensity profile requires at least two points.")
        object.__setattr__(self, "points", points)
        _require_positive("line_width", self.line_width)
        _require_positive("sample_spacing", self.sample_spacing)
        _require_positive("pixel_size_x", self.pixel_size_x)
        _require_positive("pixel_size_y", self.pixel_size_y)
        object.__setattr__(self, "line_width", float(self.line_width))
        object.__setattr__(self, "sample_spacing", float(self.sample_spacing))
        object.__setattr__(self, "pixel_size_x", float(self.pixel_size_x))
        object.__setattr__(self, "pixel_size_y", float(self.pixel_size_y))
        object.__setattr__(self, "channel", str(self.channel))
        object.__setattr__(self, "request_id", str(self.request_id))
        object.__setattr__(self, "generation", int(self.generation))


@dataclass(frozen=True, slots=True)
class IntensityProfileResult:
    distances_px: tuple[float, ...]
    distances: tuple[float, ...]
    values: tuple[float | None, ...]
    sample_points_px: tuple[Coordinate, ...]
    valid_sample_count: int
    channel: str
    request_id: str = ""
    generation: int = 0


@dataclass(frozen=True, slots=True)
class ParticleAnalysisRequest:
    mask: NDArray[np.bool_]
    connectivity: int = 8
    min_area_px: int = 1
    max_area_px: int | None = None
    min_circularity: float = 0.0
    max_circularity: float = 1.0
    include_holes: bool = False
    exclude_edge: bool = False
    pixel_size_x: float = 1.0
    pixel_size_y: float = 1.0
    unit: str = "px"
    request_id: str = ""
    generation: int = 0

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask, dtype=bool)
        if mask.ndim != 2 or mask.shape[0] <= 0 or mask.shape[1] <= 0:
            raise ValueError("particle mask must be a non-empty H×W array.")
        frozen = np.ascontiguousarray(mask).copy()
        frozen.setflags(write=False)
        object.__setattr__(self, "mask", frozen)
        if int(self.connectivity) not in {4, 8}:
            raise ValueError("particle connectivity must be 4 or 8.")
        if int(self.min_area_px) < 1:
            raise ValueError("min_area_px must be at least 1.")
        if self.max_area_px is not None and int(self.max_area_px) < int(self.min_area_px):
            raise ValueError("max_area_px must be greater than or equal to min_area_px.")
        minimum_circularity = float(self.min_circularity)
        maximum_circularity = float(self.max_circularity)
        if not 0.0 <= minimum_circularity <= maximum_circularity <= 1.0:
            raise ValueError("circularity limits must satisfy 0 <= min <= max <= 1.")
        _require_positive("pixel_size_x", self.pixel_size_x)
        _require_positive("pixel_size_y", self.pixel_size_y)
        object.__setattr__(self, "connectivity", int(self.connectivity))
        object.__setattr__(self, "min_area_px", int(self.min_area_px))
        object.__setattr__(
            self,
            "max_area_px",
            None if self.max_area_px is None else int(self.max_area_px),
        )
        object.__setattr__(self, "min_circularity", minimum_circularity)
        object.__setattr__(self, "max_circularity", maximum_circularity)
        object.__setattr__(self, "pixel_size_x", float(self.pixel_size_x))
        object.__setattr__(self, "pixel_size_y", float(self.pixel_size_y))
        object.__setattr__(self, "unit", str(self.unit or "px"))
        object.__setattr__(self, "request_id", str(self.request_id))
        object.__setattr__(self, "generation", int(self.generation))


@dataclass(frozen=True, slots=True)
class Particle:
    index: int
    exact_area_px: int
    area: float
    centroid_px: Coordinate
    bounds_px: tuple[int, int, int, int]
    rings: ImmutableRings
    perimeter_px: float
    circularity: float | None
    hole_count: int
    touches_edge: bool


@dataclass(frozen=True, slots=True)
class ParticleAnalysisResult:
    particles: tuple[Particle, ...]
    total_component_count: int
    accepted_count: int
    rejected_by_area_count: int
    rejected_by_circularity_count: int
    rejected_edge_count: int
    foreground_pixel_count: int
    include_holes: bool
    connectivity: int
    request_id: str = ""
    generation: int = 0


@dataclass(frozen=True, slots=True)
class FindMaximaRequest:
    image: NDArray[Any]
    roi_mask: NDArray[np.bool_] | None = None
    channel: str = "luminance"
    minimum_value: float | None = None
    prominence: float = 0.0
    neighborhood_radius: int = 1
    min_distance: float = 1.0
    exclude_edge: bool = False
    max_points: int | None = None
    request_id: str = ""
    generation: int = 0

    def __post_init__(self) -> None:
        image = _freeze_image(self.image)
        object.__setattr__(self, "image", image)
        object.__setattr__(
            self,
            "roi_mask",
            None
            if self.roi_mask is None
            else _freeze_mask(self.roi_mask, image.shape[:2]),
        )
        if self.minimum_value is not None and not math.isfinite(float(self.minimum_value)):
            raise ValueError("minimum_value must be finite.")
        if not math.isfinite(float(self.prominence)) or float(self.prominence) < 0:
            raise ValueError("prominence must be finite and non-negative.")
        if int(self.neighborhood_radius) < 1:
            raise ValueError("neighborhood_radius must be at least 1.")
        _require_positive("min_distance", self.min_distance)
        if self.max_points is not None and int(self.max_points) < 1:
            raise ValueError("max_points must be at least 1.")
        object.__setattr__(
            self,
            "minimum_value",
            None if self.minimum_value is None else float(self.minimum_value),
        )
        object.__setattr__(self, "prominence", float(self.prominence))
        object.__setattr__(self, "neighborhood_radius", int(self.neighborhood_radius))
        object.__setattr__(self, "min_distance", float(self.min_distance))
        object.__setattr__(
            self,
            "max_points",
            None if self.max_points is None else int(self.max_points),
        )
        object.__setattr__(self, "channel", str(self.channel))
        object.__setattr__(self, "request_id", str(self.request_id))
        object.__setattr__(self, "generation", int(self.generation))


@dataclass(frozen=True, slots=True)
class LocalMaximum:
    x: float
    y: float
    value: float
    local_prominence: float


@dataclass(frozen=True, slots=True)
class FindMaximaResult:
    maxima: tuple[LocalMaximum, ...]
    candidate_plateau_count: int
    suppressed_count: int
    channel: str
    request_id: str = ""
    generation: int = 0


def analyze_shape(request: ShapeAnalysisRequest) -> ShapeAnalysisResult:
    if not request.rings or len(request.rings[0]) < 3:
        raise ValueError("shape analysis requires an outer ring with at least three points.")
    point_rings = [
        [Point(float(x), float(y)) for x, y in ring]
        for ring in request.rings
        if len(ring) >= 3
    ]
    vector_area_px, vector_centroid = area_rings_area_and_centroid(point_rings)
    area_px = (
        float(request.exact_area_px)
        if request.exact_area_px is not None
        else float(vector_area_px)
    )
    area_scale = request.pixel_size_x * request.pixel_size_y
    area = area_px * area_scale
    outer = request.rings[0]
    holes = tuple(ring for ring in request.rings[1:] if len(ring) >= 3)
    outer_perimeter_px = _ring_perimeter(outer)
    hole_perimeter_px = sum(_ring_perimeter(ring) for ring in holes)
    outer_perimeter = _ring_perimeter(
        tuple(
            (x * request.pixel_size_x, y * request.pixel_size_y)
            for x, y in outer
        )
    )
    hole_perimeter = sum(
        _ring_perimeter(
            tuple(
                (x * request.pixel_size_x, y * request.pixel_size_y)
                for x, y in ring
            )
        )
        for ring in holes
    )
    total_perimeter_px = outer_perimeter_px + hole_perimeter_px
    total_perimeter = outer_perimeter + hole_perimeter
    flat_points = tuple(point for ring in request.rings for point in ring)
    xs = tuple(point[0] for point in flat_points)
    ys = tuple(point[1] for point in flat_points)
    bounds = (min(xs), min(ys), max(xs), max(ys))
    hole_area_px = sum(abs(_signed_ring_area(ring)) for ring in holes)
    physical_outer = np.asarray(
        [
            (x * request.pixel_size_x, y * request.pixel_size_y)
            for x, y in outer
        ],
        dtype=np.float64,
    )
    hull = _convex_hull(physical_outer)
    convex_area = abs(_signed_ring_area(tuple(map(tuple, hull.tolist())))) if len(hull) >= 3 else 0.0
    feret_max, feret_angle = _maximum_feret(hull)
    feret_min = _minimum_feret(hull)
    ellipse_major, ellipse_minor, ellipse_angle = _fit_ellipse(physical_outer)
    if ellipse_major is None or ellipse_minor is None:
        minimum_rectangle = cv2.minAreaRect(physical_outer.astype(np.float32))
        width, height = (float(minimum_rectangle[1][0]), float(minimum_rectangle[1][1]))
        if width > 0 and height > 0:
            ellipse_major = max(width, height)
            ellipse_minor = min(width, height)
            ellipse_angle = float(minimum_rectangle[2]) % 180.0
    equivalent = math.sqrt(4.0 * area / math.pi) if area > 0 else None
    circularity = (
        4.0 * math.pi * area / (total_perimeter * total_perimeter)
        if area > 0 and total_perimeter > 0
        else None
    )
    if circularity is not None:
        circularity = min(1.0, max(0.0, circularity))
    aspect_ratio = (
        ellipse_major / ellipse_minor
        if ellipse_major is not None and ellipse_minor is not None and ellipse_minor > 0
        else None
    )
    roundness = (
        4.0 * area / (math.pi * ellipse_major * ellipse_major)
        if area > 0 and ellipse_major is not None and ellipse_major > 0
        else None
    )
    solidity = area / convex_area if area > 0 and convex_area > 0 else None
    warnings: list[str] = []
    if solidity is not None and solidity > 1.0 + 1e-6:
        warnings.append("精确掩膜面积大于矢量凸包面积；实心度保留原始比值供复核。")
    centroid = (
        vector_centroid.x * request.pixel_size_x,
        vector_centroid.y * request.pixel_size_y,
    )
    return ShapeAnalysisResult(
        area_px=area_px,
        vector_area_px=float(vector_area_px),
        area=area,
        centroid_px=(float(vector_centroid.x), float(vector_centroid.y)),
        centroid=centroid,
        outer_perimeter_px=outer_perimeter_px,
        hole_perimeter_px=hole_perimeter_px,
        total_perimeter_px=total_perimeter_px,
        outer_perimeter=outer_perimeter,
        hole_perimeter=hole_perimeter,
        total_perimeter=total_perimeter,
        bounds_px=bounds,
        hole_count=len(holes),
        hole_area_px=hole_area_px,
        equivalent_circle_diameter=equivalent,
        feret_max=feret_max,
        feret_min=feret_min,
        feret_angle_degrees=feret_angle,
        ellipse_major=ellipse_major,
        ellipse_minor=ellipse_minor,
        ellipse_angle_degrees=ellipse_angle,
        circularity=circularity,
        aspect_ratio=aspect_ratio,
        roundness=roundness,
        solidity=solidity,
        unit=request.unit,
        area_from_exact_mask=request.exact_area_px is not None,
        warnings=tuple(warnings),
        request_id=request.request_id,
        generation=request.generation,
    )


def analyze_intensity(request: IntensityAnalysisRequest) -> IntensityAnalysisResult:
    scalar = _select_scalar_channel(request.image, request.channel).astype(np.float64)
    mask = _resolve_analysis_mask(
        scalar.shape,
        roi_mask=request.roi_mask,
        rings=request.rings,
    )
    selected = scalar[mask]
    included_count = int(selected.size)
    finite = np.isfinite(selected)
    values = selected[finite]
    valid_count = int(values.size)
    non_finite_count = included_count - valid_count
    if valid_count == 0:
        return IntensityAnalysisResult(
            included_pixel_count=included_count,
            valid_pixel_count=0,
            non_finite_count=non_finite_count,
            mean=None,
            median=None,
            stddev=None,
            minimum=None,
            maximum=None,
            integrated_density=None,
            intensity_centroid_px=None,
            percentiles=(),
            channel=request.channel,
            request_id=request.request_id,
            generation=request.generation,
        )
    mean = float(np.mean(values))
    median = float(np.median(values))
    stddev = float(np.std(values, ddof=0))
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    integrated_density = float(np.sum(values, dtype=np.float64))
    y_coords, x_coords = np.nonzero(mask & np.isfinite(scalar))
    weights = scalar[y_coords, x_coords].astype(np.float64)
    total_weight = float(np.sum(weights, dtype=np.float64))
    intensity_centroid = (
        (
            float(np.sum(x_coords * weights, dtype=np.float64) / total_weight),
            float(np.sum(y_coords * weights, dtype=np.float64) / total_weight),
        )
        if math.isfinite(total_weight) and not math.isclose(total_weight, 0.0)
        else None
    )
    percentile_values = np.percentile(values, request.percentile_levels)
    return IntensityAnalysisResult(
        included_pixel_count=included_count,
        valid_pixel_count=valid_count,
        non_finite_count=non_finite_count,
        mean=mean,
        median=median,
        stddev=stddev,
        minimum=minimum,
        maximum=maximum,
        integrated_density=integrated_density,
        intensity_centroid_px=intensity_centroid,
        percentiles=tuple(
            (float(level), float(value))
            for level, value in zip(
                request.percentile_levels,
                percentile_values,
                strict=True,
            )
        ),
        channel=request.channel,
        request_id=request.request_id,
        generation=request.generation,
    )


def calculate_histogram(request: HistogramRequest) -> HistogramResult:
    scalar = _select_scalar_channel(request.image, request.channel).astype(np.float64)
    mask = _resolve_analysis_mask(
        scalar.shape,
        roi_mask=request.roi_mask,
        rings=request.rings,
    )
    selected = scalar[mask]
    finite = np.isfinite(selected)
    values = selected[finite]
    if request.value_range is None:
        if values.size:
            low = float(np.min(values))
            high = float(np.max(values))
            if math.isclose(low, high):
                low -= 0.5
                high += 0.5
        else:
            low, high = 0.0, 1.0
        value_range = (low, high)
    else:
        value_range = request.value_range
    counts, edges = np.histogram(values, bins=request.bins, range=value_range)
    return HistogramResult(
        counts=tuple(int(value) for value in counts),
        edges=tuple(float(value) for value in edges),
        included_pixel_count=int(selected.size),
        non_finite_count=int(selected.size - values.size),
        channel=request.channel,
        request_id=request.request_id,
        generation=request.generation,
    )


def sample_intensity_profile(request: IntensityProfileRequest) -> IntensityProfileResult:
    scalar = _select_scalar_channel(request.image, request.channel).astype(np.float64)
    points = request.points
    segment_lengths = [
        math.hypot(points[index + 1][0] - points[index][0], points[index + 1][1] - points[index][1])
        for index in range(len(points) - 1)
    ]
    total_length = sum(segment_lengths)
    sample_count = max(2, int(math.floor(total_length / request.sample_spacing)) + 1)
    target_distances = np.linspace(0.0, total_length, sample_count, dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths, dtype=np.float64)))
    width_sample_count = max(1, int(math.ceil(request.line_width)))
    offsets = (
        np.asarray([0.0], dtype=np.float64)
        if width_sample_count == 1
        else np.linspace(
            -(request.line_width - 1.0) / 2.0,
            (request.line_width - 1.0) / 2.0,
            width_sample_count,
            dtype=np.float64,
        )
    )
    sample_points: list[Coordinate] = []
    values: list[float | None] = []
    physical_distances: list[float] = []
    physical_running = 0.0
    physical_segment_lengths = [
        math.hypot(
            (points[index + 1][0] - points[index][0]) * request.pixel_size_x,
            (points[index + 1][1] - points[index][1]) * request.pixel_size_y,
        )
        for index in range(len(points) - 1)
    ]
    for target in target_distances:
        segment_index = min(
            len(segment_lengths) - 1,
            max(0, int(np.searchsorted(cumulative, target, side="right") - 1)),
        )
        segment_length = segment_lengths[segment_index]
        fraction = (
            0.0
            if segment_length <= 1e-12
            else float((target - cumulative[segment_index]) / segment_length)
        )
        x0, y0 = points[segment_index]
        x1, y1 = points[segment_index + 1]
        x = x0 + (x1 - x0) * fraction
        y = y0 + (y1 - y0) * fraction
        sample_points.append((float(x), float(y)))
        dx = x1 - x0
        dy = y1 - y0
        norm = math.hypot(dx, dy)
        normal_x, normal_y = ((0.0, 0.0) if norm <= 1e-12 else (-dy / norm, dx / norm))
        width_values = [
            _bilinear_sample(scalar, x + normal_x * offset, y + normal_y * offset)
            for offset in offsets
        ]
        finite_values = [value for value in width_values if value is not None]
        values.append(
            float(sum(finite_values) / len(finite_values))
            if finite_values
            else None
        )
        physical_running = sum(physical_segment_lengths[:segment_index])
        physical_running += physical_segment_lengths[segment_index] * fraction
        physical_distances.append(float(physical_running))
    return IntensityProfileResult(
        distances_px=tuple(float(value) for value in target_distances),
        distances=tuple(physical_distances),
        values=tuple(values),
        sample_points_px=tuple(sample_points),
        valid_sample_count=sum(value is not None for value in values),
        channel=request.channel,
        request_id=request.request_id,
        generation=request.generation,
    )


def analyze_particles(request: ParticleAnalysisRequest) -> ParticleAnalysisResult:
    working_mask = np.asarray(request.mask, dtype=bool)
    if request.include_holes:
        working_mask = _fill_mask_holes(working_mask, connectivity=request.connectivity)
    foreground_pixel_count = int(np.count_nonzero(working_mask))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        working_mask.astype(np.uint8),
        connectivity=request.connectivity,
    )
    accepted: list[Particle] = []
    rejected_by_area = 0
    rejected_by_circularity = 0
    rejected_edge = 0
    height, width = working_mask.shape
    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        exact_area_px = int(stats[label, cv2.CC_STAT_AREA])
        if exact_area_px < request.min_area_px or (
            request.max_area_px is not None and exact_area_px > request.max_area_px
        ):
            rejected_by_area += 1
            continue
        touches_edge = (
            x == 0
            or y == 0
            or x + component_width >= width
            or y + component_height >= height
        )
        if request.exclude_edge and touches_edge:
            rejected_edge += 1
            continue
        local = (labels[y : y + component_height, x : x + component_width] == label)
        rings = _component_rings(local, offset_x=x, offset_y=y)
        perimeter = sum(_ring_perimeter(ring) for ring in rings)
        circularity = (
            min(1.0, 4.0 * math.pi * exact_area_px / (perimeter * perimeter))
            if perimeter > 0
            else None
        )
        circularity_for_filter = 0.0 if circularity is None else circularity
        if not request.min_circularity <= circularity_for_filter <= request.max_circularity:
            rejected_by_circularity += 1
            continue
        accepted.append(
            Particle(
                index=0,
                exact_area_px=exact_area_px,
                area=exact_area_px * request.pixel_size_x * request.pixel_size_y,
                centroid_px=(
                    float(centroids[label, 0]),
                    float(centroids[label, 1]),
                ),
                bounds_px=(x, y, component_width, component_height),
                rings=rings,
                perimeter_px=perimeter,
                circularity=circularity,
                hole_count=max(0, len(rings) - 1),
                touches_edge=touches_edge,
            )
        )
    accepted.sort(key=lambda item: (item.bounds_px[1], item.bounds_px[0], item.exact_area_px))
    accepted = [
        replace(particle, index=index)
        for index, particle in enumerate(accepted, start=1)
    ]
    return ParticleAnalysisResult(
        particles=tuple(accepted),
        total_component_count=max(0, count - 1),
        accepted_count=len(accepted),
        rejected_by_area_count=rejected_by_area,
        rejected_by_circularity_count=rejected_by_circularity,
        rejected_edge_count=rejected_edge,
        foreground_pixel_count=foreground_pixel_count,
        include_holes=request.include_holes,
        connectivity=request.connectivity,
        request_id=request.request_id,
        generation=request.generation,
    )


def find_local_maxima(request: FindMaximaRequest) -> FindMaximaResult:
    source = _select_scalar_channel(request.image, request.channel).astype(np.float32)
    finite = np.isfinite(source)
    if not np.any(finite):
        return FindMaximaResult(
            maxima=(),
            candidate_plateau_count=0,
            suppressed_count=0,
            channel=request.channel,
            request_id=request.request_id,
            generation=request.generation,
        )
    finite_minimum = float(np.min(source[finite]))
    safe = np.where(finite, source, finite_minimum - 1.0).astype(np.float32)
    radius = request.neighborhood_radius
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
    local_max = cv2.dilate(safe, kernel, borderType=cv2.BORDER_REPLICATE)
    local_min = cv2.erode(safe, kernel, borderType=cv2.BORDER_REPLICATE)
    candidates = finite & np.isclose(safe, local_max, rtol=0.0, atol=1e-7)
    candidates &= (safe - local_min) >= request.prominence
    if request.minimum_value is not None:
        candidates &= safe >= request.minimum_value
    if request.roi_mask is not None:
        candidates &= request.roi_mask
    if request.exclude_edge:
        candidates[:radius, :] = False
        candidates[-radius:, :] = False
        candidates[:, :radius] = False
        candidates[:, -radius:] = False
    plateau_count, plateau_labels = cv2.connectedComponents(
        candidates.astype(np.uint8),
        connectivity=8,
    )
    plateau_maxima: list[LocalMaximum] = []
    for label in range(1, plateau_count):
        ys, xs = np.nonzero(plateau_labels == label)
        if xs.size == 0:
            continue
        values = safe[ys, xs]
        best_value = float(np.max(values))
        best_indices = np.flatnonzero(values == best_value)
        best_index = min(
            (int(index) for index in best_indices),
            key=lambda index: (int(ys[index]), int(xs[index])),
        )
        y = int(ys[best_index])
        x = int(xs[best_index])
        plateau_maxima.append(
            LocalMaximum(
                x=float(x),
                y=float(y),
                value=best_value,
                local_prominence=float(best_value - local_min[y, x]),
            )
        )
    plateau_maxima.sort(key=lambda item: (-item.value, item.y, item.x))
    accepted: list[LocalMaximum] = []
    minimum_distance_squared = request.min_distance * request.min_distance
    for candidate in plateau_maxima:
        if any(
            (candidate.x - existing.x) ** 2 + (candidate.y - existing.y) ** 2
            < minimum_distance_squared
            for existing in accepted
        ):
            continue
        accepted.append(candidate)
        if request.max_points is not None and len(accepted) >= request.max_points:
            break
    return FindMaximaResult(
        maxima=tuple(accepted),
        candidate_plateau_count=max(0, plateau_count - 1),
        suppressed_count=max(0, len(plateau_maxima) - len(accepted)),
        channel=request.channel,
        request_id=request.request_id,
        generation=request.generation,
    )


def _freeze_image(image: NDArray[Any]) -> NDArray[Any]:
    array = np.asarray(image)
    if array.ndim not in {2, 3} or array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError("analysis image must have shape H×W or H×W×C.")
    if array.ndim == 3 and array.shape[2] not in {1, 3, 4}:
        raise ValueError("analysis image channel count must be 1, 3 or 4.")
    if array.dtype not in {
        np.dtype(np.uint8),
        np.dtype(np.uint16),
        np.dtype(np.float32),
    }:
        raise TypeError("analysis image dtype must be uint8, uint16 or float32.")
    frozen = np.ascontiguousarray(array).copy()
    frozen.setflags(write=False)
    return frozen


def _freeze_mask(mask: NDArray[np.bool_], shape: tuple[int, int]) -> NDArray[np.bool_]:
    array = np.asarray(mask, dtype=bool)
    if array.shape != shape:
        raise ValueError(f"analysis mask shape {array.shape!r} does not match {shape!r}.")
    frozen = np.ascontiguousarray(array).copy()
    frozen.setflags(write=False)
    return frozen


def _freeze_rings(rings: Iterable[Iterable[Any]]) -> ImmutableRings:
    return tuple(_freeze_ring(ring) for ring in rings)


def _freeze_ring(ring: Iterable[Any]) -> ImmutableRing:
    coordinates: list[Coordinate] = []
    for point in ring:
        if hasattr(point, "x") and hasattr(point, "y"):
            x = float(point.x)
            y = float(point.y)
        else:
            x = float(point[0])
            y = float(point[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("geometry coordinates must be finite.")
        coordinates.append((x, y))
    return tuple(coordinates)


def _select_scalar_channel(image: NDArray[Any], channel: str) -> NDArray[Any]:
    if image.ndim == 2:
        return image
    if image.shape[2] == 1:
        return image[..., 0]
    resolved = str(channel).strip().lower()
    index = {"red": 0, "r": 0, "green": 1, "g": 1, "blue": 2, "b": 2}.get(resolved)
    if index is not None:
        return image[..., index]
    if resolved in {"luminance", "gray", "grayscale"}:
        rgb = image[..., :3].astype(np.float64)
        return (
            rgb[..., 0] * 0.2126
            + rgb[..., 1] * 0.7152
            + rgb[..., 2] * 0.0722
        ).astype(np.float32)
    raise ValueError(f"Unsupported analysis channel: {channel}")


def _resolve_analysis_mask(
    shape: tuple[int, int],
    *,
    roi_mask: NDArray[np.bool_] | None,
    rings: ImmutableRings,
) -> NDArray[np.bool_]:
    result = np.ones(shape, dtype=bool)
    if rings:
        result &= _rings_to_odd_even_mask(shape, rings)
    if roi_mask is not None:
        result &= roi_mask
    return result


def _rings_to_odd_even_mask(
    shape: tuple[int, int],
    rings: ImmutableRings,
) -> NDArray[np.bool_]:
    mask = np.zeros(shape, dtype=bool)
    for ring in rings:
        if len(ring) < 3:
            continue
        contour = np.rint(np.asarray(ring, dtype=np.float64)).astype(np.int32)
        temporary = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(temporary, [contour], 1)
        mask ^= temporary.astype(bool)
    return mask


def _ring_perimeter(ring: ImmutableRing) -> float:
    if len(ring) < 2:
        return 0.0
    return sum(
        math.hypot(
            ring[(index + 1) % len(ring)][0] - ring[index][0],
            ring[(index + 1) % len(ring)][1] - ring[index][1],
        )
        for index in range(len(ring))
    )


def _signed_ring_area(ring: ImmutableRing) -> float:
    if len(ring) < 3:
        return 0.0
    return 0.5 * sum(
        ring[index][0] * ring[(index + 1) % len(ring)][1]
        - ring[(index + 1) % len(ring)][0] * ring[index][1]
        for index in range(len(ring))
    )


def _convex_hull(points: NDArray[np.float64]) -> NDArray[np.float64]:
    if len(points) < 3:
        return points.copy()
    hull = cv2.convexHull(points.astype(np.float32), returnPoints=True)
    return hull.reshape(-1, 2).astype(np.float64)


def _maximum_feret(hull: NDArray[np.float64]) -> tuple[float | None, float | None]:
    if len(hull) < 2:
        return None, None
    maximum_squared = -1.0
    best: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None
    for index in range(len(hull) - 1):
        differences = hull[index + 1 :] - hull[index]
        distances = np.einsum("ij,ij->i", differences, differences)
        relative = int(np.argmax(distances))
        value = float(distances[relative])
        if value > maximum_squared:
            maximum_squared = value
            best = (hull[index], hull[index + 1 + relative])
    if best is None:
        return None, None
    vector = best[1] - best[0]
    angle = math.degrees(math.atan2(-float(vector[1]), float(vector[0]))) % 180.0
    return math.sqrt(maximum_squared), angle


def _minimum_feret(hull: NDArray[np.float64]) -> float | None:
    if len(hull) < 3:
        return 0.0 if len(hull) == 2 else None
    minimum_width = math.inf
    for index in range(len(hull)):
        start = hull[index]
        end = hull[(index + 1) % len(hull)]
        edge = end - start
        length = float(np.linalg.norm(edge))
        if length <= 1e-12:
            continue
        distances = np.abs(
            edge[0] * (start[1] - hull[:, 1])
            - (start[0] - hull[:, 0]) * edge[1]
        ) / length
        minimum_width = min(minimum_width, float(np.max(distances)))
    return minimum_width if math.isfinite(minimum_width) else None


def _fit_ellipse(
    points: NDArray[np.float64],
) -> tuple[float | None, float | None, float | None]:
    if len(points) < 5:
        return None, None, None
    try:
        _center, axes, angle = cv2.fitEllipse(points.astype(np.float32))
    except cv2.error:
        return None, None, None
    major = max(float(axes[0]), float(axes[1]))
    minor = min(float(axes[0]), float(axes[1]))
    if major <= 0 or minor <= 0:
        return None, None, None
    resolved_angle = float(angle)
    if axes[0] < axes[1]:
        resolved_angle += 90.0
    return major, minor, resolved_angle % 180.0


def _bilinear_sample(
    image: NDArray[np.float64],
    x: float,
    y: float,
) -> float | None:
    height, width = image.shape
    if x < 0 or y < 0 or x > width - 1 or y > height - 1:
        return None
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(width - 1, x0 + 1)
    y1 = min(height - 1, y0 + 1)
    dx = x - x0
    dy = y - y0
    samples = (
        (image[y0, x0], (1.0 - dx) * (1.0 - dy)),
        (image[y0, x1], dx * (1.0 - dy)),
        (image[y1, x0], (1.0 - dx) * dy),
        (image[y1, x1], dx * dy),
    )
    finite_samples = [(float(value), weight) for value, weight in samples if math.isfinite(float(value))]
    weight_sum = sum(weight for _value, weight in finite_samples)
    if weight_sum <= 0:
        return None
    return sum(value * weight for value, weight in finite_samples) / weight_sum


def _fill_mask_holes(
    mask: NDArray[np.bool_],
    *,
    connectivity: int,
) -> NDArray[np.bool_]:
    inverse = (~mask).astype(np.uint8)
    count, labels = cv2.connectedComponents(inverse, connectivity=connectivity)
    border_labels = set(
        np.concatenate((labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1])).tolist()
    )
    filled = mask.copy()
    for label in range(1, count):
        if label not in border_labels:
            filled[labels == label] = True
    return filled


def _component_rings(
    local_mask: NDArray[np.bool_],
    *,
    offset_x: int,
    offset_y: int,
) -> ImmutableRings:
    contours, hierarchy = cv2.findContours(
        local_mask.astype(np.uint8),
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours or hierarchy is None:
        return ()
    hierarchy_values = hierarchy[0]
    external_indices = [
        index for index, record in enumerate(hierarchy_values) if int(record[3]) < 0
    ]
    if not external_indices:
        return ()
    outer_index = max(external_indices, key=lambda index: abs(cv2.contourArea(contours[index])))
    ordered_indices = [outer_index]
    child = int(hierarchy_values[outer_index][2])
    while child >= 0:
        ordered_indices.append(child)
        child = int(hierarchy_values[child][0])
    rings: list[ImmutableRing] = []
    for index in ordered_indices:
        points = contours[index].reshape(-1, 2)
        if len(points) < 3:
            continue
        rings.append(
            tuple(
                (float(x + offset_x), float(y + offset_y))
                for x, y in points
            )
        )
    return tuple(rings)


def _require_positive(name: str, value: float) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{name} must be a finite positive number.")


class ImageAnalysisService:
    """Stateless façade for worker-side analysis dispatch."""

    analyze_shape = staticmethod(analyze_shape)
    analyze_intensity = staticmethod(analyze_intensity)
    calculate_histogram = staticmethod(calculate_histogram)
    sample_intensity_profile = staticmethod(sample_intensity_profile)
    analyze_particles = staticmethod(analyze_particles)
    find_local_maxima = staticmethod(find_local_maxima)
