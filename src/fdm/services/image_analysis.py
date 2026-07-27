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
from fdm.services.image_processing import fft_power_spectrum


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
class ShapeComponentResult:
    index: int
    area_px: float
    area: float
    centroid_px: Coordinate
    centroid: Coordinate
    outer_perimeter_px: float
    hole_perimeter_px: float
    total_perimeter_px: float
    hole_count: int
    bounds_px: tuple[float, float, float, float]
    extent: float | None
    convex_area: float | None
    solidity: float | None


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
    component_count: int
    hole_count: int
    euler_number: int
    hole_area_px: float
    extent: float | None
    component_table: tuple[ShapeComponentResult, ...]
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

    @property
    def euler(self) -> int:
        return self.euler_number

    @property
    def global_solidity(self) -> float | None:
        return self.solidity

    @property
    def components(self) -> tuple[ShapeComponentResult, ...]:
        return self.component_table


@dataclass(frozen=True, slots=True)
class IntensityAnalysisRequest:
    image: NDArray[Any]
    roi_mask: NDArray[np.bool_] | None = None
    rings: ImmutableRings = ()
    channel: str = "luminance"
    percentile_levels: tuple[float, ...] = (10.0, 25.0, 50.0, 75.0, 90.0)
    threshold_low: float | None = None
    threshold_high: float | None = None
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
        low = _optional_finite(self.threshold_low, "threshold_low")
        high = _optional_finite(self.threshold_high, "threshold_high")
        if low is not None and high is not None and high < low:
            raise ValueError("threshold_high must be greater than or equal to threshold_low.")
        object.__setattr__(self, "threshold_low", low)
        object.__setattr__(self, "threshold_high", high)
        object.__setattr__(self, "channel", str(self.channel).strip().lower())
        object.__setattr__(self, "request_id", str(self.request_id))
        object.__setattr__(self, "generation", int(self.generation))


@dataclass(frozen=True, slots=True)
class IntensityChannelStatistics:
    channel: str
    valid_pixel_count: int
    mean: float | None
    median: float | None
    mode: float | None
    stddev: float | None
    skewness: float | None
    excess_kurtosis: float | None
    minimum: float | None
    maximum: float | None
    integrated_density: float | None
    threshold_area_fraction: float | None


@dataclass(frozen=True, slots=True)
class IntensityAnalysisResult:
    included_pixel_count: int
    valid_pixel_count: int
    non_finite_count: int
    mean: float | None
    median: float | None
    mode: float | None
    stddev: float | None
    skewness: float | None
    excess_kurtosis: float | None
    minimum: float | None
    maximum: float | None
    integrated_density: float | None
    threshold_area_fraction: float | None
    intensity_centroid_px: Coordinate | None
    percentiles: tuple[tuple[float, float], ...]
    channel_statistics: tuple[IntensityChannelStatistics, ...]
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
    log_counts: bool = False
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
        object.__setattr__(self, "log_counts", bool(self.log_counts))
        object.__setattr__(self, "request_id", str(self.request_id))
        object.__setattr__(self, "generation", int(self.generation))


@dataclass(frozen=True, slots=True)
class HistogramResult:
    counts: tuple[int, ...]
    display_counts: tuple[float, ...]
    edges: tuple[float, ...]
    included_pixel_count: int
    non_finite_count: int
    channel: str
    log_counts: bool
    request_id: str = ""
    generation: int = 0


@dataclass(frozen=True, slots=True)
class FftPowerSpectrumRequest:
    """Frozen source and region used to create one auditable FFT asset."""

    image: NDArray[Any]
    roi_mask: NDArray[np.bool_] | None = None
    rings: ImmutableRings = ()
    channel: str = "luminance"
    logarithmic: bool = True
    centered: bool = True
    window: str = "none"
    tukey_alpha: float = 0.25
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
        object.__setattr__(self, "channel", str(self.channel).strip().lower())
        object.__setattr__(self, "logarithmic", bool(self.logarithmic))
        object.__setattr__(self, "centered", bool(self.centered))
        window = str(self.window).strip().lower()
        if window not in {"none", "tukey"}:
            raise ValueError("FFT 功率谱窗函数必须为 none 或 tukey。")
        alpha = float(self.tukey_alpha)
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("Tukey alpha 必须是 0 到 1 之间的有限数。")
        object.__setattr__(self, "window", window)
        object.__setattr__(self, "tukey_alpha", alpha)
        object.__setattr__(self, "request_id", str(self.request_id))
        object.__setattr__(self, "generation", int(self.generation))


@dataclass(frozen=True, slots=True)
class FftPowerSpectrumResult:
    power: NDArray[np.float32]
    source_size: tuple[int, int]
    analysis_bounds: tuple[int, int, int, int]
    roi_applied: bool
    mask_policy: str
    channel: str
    logarithmic: bool
    centered: bool
    window: str
    tukey_alpha: float
    request_id: str = ""
    generation: int = 0

    def __post_init__(self) -> None:
        power = np.asarray(self.power, dtype=np.float32)
        if power.ndim != 2 or power.shape[0] <= 0 or power.shape[1] <= 0:
            raise ValueError("FFT 功率谱必须是非空二维数组。")
        frozen = np.ascontiguousarray(power).copy()
        frozen.setflags(write=False)
        object.__setattr__(self, "power", frozen)
        source_width, source_height = (
            int(self.source_size[0]),
            int(self.source_size[1]),
        )
        if source_width <= 0 or source_height <= 0:
            raise ValueError("FFT 源图尺寸必须为正数。")
        x, y, width, height = (int(value) for value in self.analysis_bounds)
        if (
            x < 0
            or y < 0
            or width <= 0
            or height <= 0
            or x + width > source_width
            or y + height > source_height
            or frozen.shape != (height, width)
        ):
            raise ValueError("FFT 分析范围与功率谱尺寸不匹配。")
        object.__setattr__(self, "source_size", (source_width, source_height))
        object.__setattr__(self, "analysis_bounds", (x, y, width, height))
        object.__setattr__(self, "roi_applied", bool(self.roi_applied))
        object.__setattr__(self, "mask_policy", str(self.mask_policy))
        object.__setattr__(self, "channel", str(self.channel))
        object.__setattr__(self, "logarithmic", bool(self.logarithmic))
        object.__setattr__(self, "centered", bool(self.centered))
        object.__setattr__(self, "window", str(self.window))
        object.__setattr__(self, "tukey_alpha", float(self.tukey_alpha))
        object.__setattr__(self, "request_id", str(self.request_id))
        object.__setattr__(self, "generation", int(self.generation))


@dataclass(frozen=True, slots=True)
class IntensityProfileRequest:
    image: NDArray[Any]
    points: tuple[Coordinate, ...]
    line_width: float = 1.0
    sample_spacing: float = 1.0
    pixel_size_x: float = 1.0
    pixel_size_y: float = 1.0
    channel: str = "luminance"
    aggregation: str = "line"
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
        aggregation = str(self.aggregation).strip().lower()
        if aggregation not in {"line", "rectangle_rows", "rectangle_columns"}:
            raise ValueError(
                "profile aggregation must be line, rectangle_rows or rectangle_columns."
            )
        object.__setattr__(self, "aggregation", aggregation)
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
    aggregation: str
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
    watershed: bool = False
    watershed_min_distance: int = 3
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
        if int(self.watershed_min_distance) < 1:
            raise ValueError("watershed_min_distance must be at least 1.")
        object.__setattr__(self, "watershed", bool(self.watershed))
        object.__setattr__(
            self,
            "watershed_min_distance",
            int(self.watershed_min_distance),
        )
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
    accepted_foreground_pixel_count: int
    area_fraction: float
    area_summary: tuple[tuple[str, float | None], ...]
    label_image: NDArray[np.int32]
    contour_image: NDArray[np.bool_]
    include_holes: bool
    connectivity: int
    watershed: bool
    request_id: str = ""
    generation: int = 0

    def __post_init__(self) -> None:
        labels = np.ascontiguousarray(self.label_image, dtype=np.int32).copy()
        contours = np.ascontiguousarray(self.contour_image, dtype=bool).copy()
        labels.setflags(write=False)
        contours.setflags(write=False)
        object.__setattr__(self, "label_image", labels)
        object.__setattr__(self, "contour_image", contours)


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
    algorithm_version: str = "1"
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
        version = str(self.algorithm_version).strip()
        if version not in {"1", "2"}:
            raise ValueError("maxima algorithm_version must be 1 or 2.")
        object.__setattr__(self, "algorithm_version", version)
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
    algorithm_version: str
    request_id: str = ""
    generation: int = 0


def analyze_shape(request: ShapeAnalysisRequest) -> ShapeAnalysisResult:
    valid_rings = tuple(ring for ring in request.rings if len(ring) >= 3)
    if not valid_rings:
        raise ValueError("shape analysis requires at least one ring with three points.")
    point_rings = [
        [Point(float(x), float(y)) for x, y in ring]
        for ring in valid_rings
    ]
    vector_area_px, vector_centroid = area_rings_area_and_centroid(point_rings)
    area_px = (
        float(request.exact_area_px)
        if request.exact_area_px is not None
        else float(vector_area_px)
    )
    area_scale = request.pixel_size_x * request.pixel_size_y
    area = area_px * area_scale
    vector_area = float(vector_area_px) * area_scale
    topology = _classify_odd_even_rings(valid_rings)
    outer_rings = tuple(
        valid_rings[index]
        for index, depth in enumerate(topology.depths)
        if depth % 2 == 0
    )
    holes = tuple(
        valid_rings[index]
        for index, depth in enumerate(topology.depths)
        if depth % 2 == 1
    )
    outer_perimeter_px = sum(_ring_perimeter(ring) for ring in outer_rings)
    hole_perimeter_px = sum(_ring_perimeter(ring) for ring in holes)
    outer_perimeter = sum(
        _ring_perimeter(
            tuple(
                (x * request.pixel_size_x, y * request.pixel_size_y)
                for x, y in ring
            )
        )
        for ring in outer_rings
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
    flat_points = tuple(point for ring in valid_rings for point in ring)
    xs = tuple(point[0] for point in flat_points)
    ys = tuple(point[1] for point in flat_points)
    bounds = (min(xs), min(ys), max(xs), max(ys))
    hole_area_px = sum(abs(_signed_ring_area(ring)) for ring in holes)
    physical_outer = np.asarray(
        [
            (x * request.pixel_size_x, y * request.pixel_size_y)
            for ring in outer_rings
            for x, y in ring
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
    equivalent = (
        math.sqrt(4.0 * vector_area / math.pi)
        if vector_area > 0
        else None
    )
    circularity = (
        4.0 * math.pi * vector_area / (total_perimeter * total_perimeter)
        if vector_area > 0 and total_perimeter > 0
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
        4.0 * vector_area / (math.pi * ellipse_major * ellipse_major)
        if vector_area > 0 and ellipse_major is not None and ellipse_major > 0
        else None
    )
    solidity = (
        vector_area / convex_area
        if vector_area > 0 and convex_area > 0
        else None
    )
    bounds_area = (
        (bounds[2] - bounds[0])
        * request.pixel_size_x
        * (bounds[3] - bounds[1])
        * request.pixel_size_y
    )
    extent = vector_area / bounds_area if bounds_area > 0 else None
    component_table = _shape_component_table(
        valid_rings,
        topology,
        pixel_size_x=request.pixel_size_x,
        pixel_size_y=request.pixel_size_y,
    )
    centroid = (
        vector_centroid.x * request.pixel_size_x,
        vector_centroid.y * request.pixel_size_y,
    )
    warnings: tuple[str, ...] = ()
    if request.exact_area_px is not None and not math.isclose(
        area_px,
        float(vector_area_px),
        rel_tol=1e-9,
        abs_tol=1e-6,
    ):
        warnings = (
            "权威面积来自精确掩膜；周长、Feret、拟合椭圆、"
            "圆度、Roundness、Solidity 与 Extent 均由 RAW rings "
            "这一套矢量几何计算，未混用精确掩膜面积。",
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
        component_count=len(component_table),
        hole_count=len(holes),
        euler_number=len(component_table) - len(holes),
        hole_area_px=hole_area_px,
        extent=extent,
        component_table=component_table,
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
        warnings=warnings,
        request_id=request.request_id,
        generation=request.generation,
    )


def analyze_intensity(request: IntensityAnalysisRequest) -> IntensityAnalysisResult:
    if request.channel == "rgb":
        if request.image.ndim != 3 or request.image.shape[2] < 3:
            raise ValueError("RGB channel statistics require an RGB or RGBA image.")
        scalar = _select_scalar_channel(request.image, "luminance").astype(np.float64)
        named_channels = (
            ("red", request.image[..., 0]),
            ("green", request.image[..., 1]),
            ("blue", request.image[..., 2]),
        )
    else:
        scalar = _select_scalar_channel(request.image, request.channel).astype(np.float64)
        named_channels = ((request.channel, scalar),)
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
    channel_statistics = tuple(
        _intensity_channel_statistics(
            name,
            np.asarray(channel_values, dtype=np.float64)[mask],
            threshold_low=request.threshold_low,
            threshold_high=request.threshold_high,
        )
        for name, channel_values in named_channels
    )
    if valid_count == 0:
        return IntensityAnalysisResult(
            included_pixel_count=included_count,
            valid_pixel_count=0,
            non_finite_count=non_finite_count,
            mean=None,
            median=None,
            mode=None,
            stddev=None,
            skewness=None,
            excess_kurtosis=None,
            minimum=None,
            maximum=None,
            integrated_density=None,
            threshold_area_fraction=None,
            intensity_centroid_px=None,
            percentiles=(),
            channel_statistics=channel_statistics,
            channel=request.channel,
            request_id=request.request_id,
            generation=request.generation,
        )
    primary = _intensity_channel_statistics(
        request.channel,
        values,
        threshold_low=request.threshold_low,
        threshold_high=request.threshold_high,
    )
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
        mean=primary.mean,
        median=primary.median,
        mode=primary.mode,
        stddev=primary.stddev,
        skewness=primary.skewness,
        excess_kurtosis=primary.excess_kurtosis,
        minimum=primary.minimum,
        maximum=primary.maximum,
        integrated_density=primary.integrated_density,
        threshold_area_fraction=primary.threshold_area_fraction,
        intensity_centroid_px=intensity_centroid,
        percentiles=tuple(
            (float(level), float(value))
            for level, value in zip(
                request.percentile_levels,
                percentile_values,
                strict=True,
            )
        ),
        channel_statistics=channel_statistics,
        channel=request.channel,
        request_id=request.request_id,
        generation=request.generation,
    )


def _intensity_channel_statistics(
    channel: str,
    selected: NDArray[np.float64],
    *,
    threshold_low: float | None,
    threshold_high: float | None,
) -> IntensityChannelStatistics:
    values = np.asarray(selected, dtype=np.float64)
    values = values[np.isfinite(values)]
    count = int(values.size)
    if count == 0:
        return IntensityChannelStatistics(
            channel=str(channel),
            valid_pixel_count=0,
            mean=None,
            median=None,
            mode=None,
            stddev=None,
            skewness=None,
            excess_kurtosis=None,
            minimum=None,
            maximum=None,
            integrated_density=None,
            threshold_area_fraction=None,
        )
    mean = float(np.mean(values))
    stddev = float(np.std(values, ddof=0))
    unique, counts = np.unique(values, return_counts=True)
    mode = float(unique[int(np.argmax(counts))])
    if stddev > 0:
        standardized = (values - mean) / stddev
        skewness = float(np.mean(standardized ** 3))
        excess_kurtosis = float(np.mean(standardized ** 4) - 3.0)
    else:
        skewness = None
        excess_kurtosis = None
    threshold_fraction = None
    if threshold_low is not None or threshold_high is not None:
        low = -math.inf if threshold_low is None else threshold_low
        high = math.inf if threshold_high is None else threshold_high
        threshold_fraction = float(np.count_nonzero((values >= low) & (values <= high)) / count)
    return IntensityChannelStatistics(
        channel=str(channel),
        valid_pixel_count=count,
        mean=mean,
        median=float(np.median(values)),
        mode=mode,
        stddev=stddev,
        skewness=skewness,
        excess_kurtosis=excess_kurtosis,
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
        integrated_density=float(np.sum(values, dtype=np.float64)),
        threshold_area_fraction=threshold_fraction,
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
        display_counts=tuple(
            float(math.log1p(value)) if request.log_counts else float(value)
            for value in counts
        ),
        edges=tuple(float(value) for value in edges),
        included_pixel_count=int(selected.size),
        non_finite_count=int(selected.size - values.size),
        channel=request.channel,
        log_counts=request.log_counts,
        request_id=request.request_id,
        generation=request.generation,
    )


def calculate_fft_power_spectrum(
    request: FftPowerSpectrumRequest,
) -> FftPowerSpectrumResult:
    """Calculate an FFT analysis asset without turning it into source pixels.

    Whole-image requests deliberately call the existing v1 numerical kernel so
    historical Process recipes and the new analysis result are byte-for-byte
    compatible for the same scalar input and parameters.  A ROI is frozen to
    its tight bounds and samples outside the exact mask are set to zero; this
    policy is persisted with the result because it affects spectral leakage.
    """

    scalar = _select_scalar_channel(request.image, request.channel)
    source_height, source_width = scalar.shape
    roi_applied = request.roi_mask is not None or bool(request.rings)
    if roi_applied:
        mask = _resolve_analysis_mask(
            scalar.shape,
            roi_mask=request.roi_mask,
            rings=request.rings,
        )
        y_indices, x_indices = np.nonzero(mask)
        if not x_indices.size:
            raise ValueError("FFT 功率谱的分析区域不包含任何像素。")
        x = int(np.min(x_indices))
        y = int(np.min(y_indices))
        right = int(np.max(x_indices)) + 1
        bottom = int(np.max(y_indices)) + 1
        cropped = np.asarray(scalar[y:bottom, x:right], dtype=np.float32)
        cropped_mask = mask[y:bottom, x:right]
        if not np.all(np.isfinite(cropped[cropped_mask])):
            raise ValueError(
                "FFT 功率谱的分析区域包含 NaN/Inf；"
                "请先使用“修复非有限值”生成派生图片。"
            )
        work = np.where(cropped_mask, cropped, np.float32(0.0))
        bounds = (x, y, right - x, bottom - y)
        mask_policy = "tight_bounds_zero_outside_exact_mask"
    else:
        if not np.all(np.isfinite(scalar)):
            raise ValueError(
                "FFT 功率谱不接受 NaN/Inf；"
                "请先使用“修复非有限值”生成派生图片。"
            )
        work = scalar
        bounds = (0, 0, source_width, source_height)
        mask_policy = "full_image"
    power = fft_power_spectrum(
        work,
        logarithmic=request.logarithmic,
        centered=request.centered,
        window=request.window,
        tukey_alpha=request.tukey_alpha,
    )
    return FftPowerSpectrumResult(
        power=power,
        source_size=(source_width, source_height),
        analysis_bounds=bounds,
        roi_applied=roi_applied,
        mask_policy=mask_policy,
        channel=request.channel,
        logarithmic=request.logarithmic,
        centered=request.centered,
        window=request.window,
        tukey_alpha=request.tukey_alpha,
        request_id=request.request_id,
        generation=request.generation,
    )


def sample_intensity_profile(request: IntensityProfileRequest) -> IntensityProfileResult:
    scalar = _select_scalar_channel(request.image, request.channel).astype(np.float64)
    if request.aggregation != "line":
        return _sample_rectangle_profile(request, scalar)
    points = request.points
    segment_lengths = [
        math.hypot(points[index + 1][0] - points[index][0], points[index + 1][1] - points[index][1])
        for index in range(len(points) - 1)
    ]
    total_length = sum(segment_lengths)
    if total_length <= 1e-12:
        target_distances = np.asarray((0.0, 0.0), dtype=np.float64)
    else:
        target_distances = np.arange(
            0.0,
            total_length,
            request.sample_spacing,
            dtype=np.float64,
        )
        if not target_distances.size or not math.isclose(
            float(target_distances[-1]),
            total_length,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            target_distances = np.append(target_distances, total_length)
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
        aggregation=request.aggregation,
        request_id=request.request_id,
        generation=request.generation,
    )


def _sample_rectangle_profile(
    request: IntensityProfileRequest,
    scalar: NDArray[np.float64],
) -> IntensityProfileResult:
    (first_x, first_y), (last_x, last_y) = request.points[0], request.points[-1]
    height, width = scalar.shape
    x0 = max(0, min(width - 1, int(math.floor(min(first_x, last_x)))))
    x1 = max(0, min(width - 1, int(math.ceil(max(first_x, last_x)))))
    y0 = max(0, min(height - 1, int(math.floor(min(first_y, last_y)))))
    y1 = max(0, min(height - 1, int(math.ceil(max(first_y, last_y)))))
    if x1 < x0 or y1 < y0:
        raise ValueError("rectangle profile does not intersect the image.")
    step = max(1, int(round(request.sample_spacing)))
    along_rows = request.aggregation == "rectangle_rows"
    indices = tuple(
        range(y0, y1 + 1, step)
        if along_rows
        else range(x0, x1 + 1, step)
    )
    values: list[float | None] = []
    sample_points: list[Coordinate] = []
    for index in indices:
        selected = (
            scalar[index, x0 : x1 + 1]
            if along_rows
            else scalar[y0 : y1 + 1, index]
        )
        finite = selected[np.isfinite(selected)]
        values.append(float(np.mean(finite)) if finite.size else None)
        sample_points.append(
            (
                float((x0 + x1) / 2.0 if along_rows else index),
                float(index if along_rows else (y0 + y1) / 2.0),
            )
        )
    origin = y0 if along_rows else x0
    distances_px = tuple(float(index - origin) for index in indices)
    scale = request.pixel_size_y if along_rows else request.pixel_size_x
    return IntensityProfileResult(
        distances_px=distances_px,
        distances=tuple(value * scale for value in distances_px),
        values=tuple(values),
        sample_points_px=tuple(sample_points),
        valid_sample_count=sum(value is not None for value in values),
        channel=request.channel,
        aggregation=request.aggregation,
        request_id=request.request_id,
        generation=request.generation,
    )


def analyze_particles(request: ParticleAnalysisRequest) -> ParticleAnalysisResult:
    working_mask = np.asarray(request.mask, dtype=bool)
    if request.include_holes:
        working_mask = _fill_mask_holes(working_mask, connectivity=request.connectivity)
    foreground_pixel_count = int(np.count_nonzero(working_mask))
    count, labels, stats, centroids = _particle_components(
        working_mask,
        connectivity=request.connectivity,
        watershed=request.watershed,
        watershed_min_distance=request.watershed_min_distance,
    )
    accepted: list[Particle] = []
    rejected_by_area = 0
    rejected_by_circularity = 0
    rejected_edge = 0
    accepted_source_labels: list[int] = []
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
        accepted_source_labels.append(label)
    ordered = sorted(
        zip(accepted, accepted_source_labels, strict=True),
        key=lambda item: (
            item[0].bounds_px[1],
            item[0].bounds_px[0],
            item[0].exact_area_px,
        ),
    )
    accepted = []
    accepted_source_labels = []
    label_image = np.zeros(labels.shape, dtype=np.int32)
    contour_image = np.zeros(labels.shape, dtype=bool)
    for index, (particle, source_label) in enumerate(ordered, start=1):
        accepted.append(replace(particle, index=index))
        accepted_source_labels.append(source_label)
        component = labels == source_label
        label_image[component] = index
        contour = cv2.morphologyEx(
            component.astype(np.uint8),
            cv2.MORPH_GRADIENT,
            np.ones((3, 3), dtype=np.uint8),
        )
        contour_image |= contour.astype(bool)
    accepted_area = sum(particle.exact_area_px for particle in accepted)
    areas = np.asarray(
        [particle.area for particle in accepted],
        dtype=np.float64,
    )
    area_summary = (
        ("minimum", float(np.min(areas)) if areas.size else None),
        ("maximum", float(np.max(areas)) if areas.size else None),
        ("mean", float(np.mean(areas)) if areas.size else None),
        ("median", float(np.median(areas)) if areas.size else None),
        ("stddev", float(np.std(areas)) if areas.size else None),
    )
    return ParticleAnalysisResult(
        particles=tuple(accepted),
        total_component_count=max(0, count - 1),
        accepted_count=len(accepted),
        rejected_by_area_count=rejected_by_area,
        rejected_by_circularity_count=rejected_by_circularity,
        rejected_edge_count=rejected_edge,
        foreground_pixel_count=foreground_pixel_count,
        accepted_foreground_pixel_count=accepted_area,
        area_fraction=(
            float(accepted_area / working_mask.size)
            if working_mask.size
            else 0.0
        ),
        area_summary=area_summary,
        label_image=label_image,
        contour_image=contour_image,
        include_holes=request.include_holes,
        connectivity=request.connectivity,
        watershed=request.watershed,
        request_id=request.request_id,
        generation=request.generation,
    )


def _particle_components(
    mask: NDArray[np.bool_],
    *,
    connectivity: int,
    watershed: bool,
    watershed_min_distance: int,
) -> tuple[int, NDArray[np.int32], NDArray[np.int32], NDArray[np.float64]]:
    if not watershed or not np.any(mask):
        return cv2.connectedComponentsWithStats(
            mask.astype(np.uint8),
            connectivity=connectivity,
        )
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    radius = max(1, int(watershed_min_distance))
    local_maximum = (
        distance
        >= cv2.dilate(
            distance,
            np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8),
        )
        - 1e-6
    ) & (distance > 0)
    seed_count, _seed_labels = cv2.connectedComponents(
        local_maximum.astype(np.uint8),
        connectivity=8,
    )
    if seed_count <= 1:
        return cv2.connectedComponentsWithStats(
            mask.astype(np.uint8),
            connectivity=connectivity,
        )
    # Marker-controlled watershed on the distance surface.  OpenCV's
    # ``distanceTransformWithLabels`` gives every foreground pixel to the
    # nearest connected peak marker, so watershed lines do not silently remove
    # source pixels from the area accounting.
    seed_surface = np.ones(mask.shape, dtype=np.uint8)
    seed_surface[local_maximum] = 0
    _nearest_distance, nearest_labels = cv2.distanceTransformWithLabels(
        seed_surface,
        cv2.DIST_L2,
        5,
        labelType=cv2.DIST_LABEL_CCOMP,
    )
    labels = np.where(mask, nearest_labels, 0).astype(np.int32)
    unique = tuple(int(value) for value in np.unique(labels) if value > 0)
    compact = np.zeros_like(labels)
    for compact_label, source_label in enumerate(unique, start=1):
        compact[labels == source_label] = compact_label
    count = len(unique) + 1
    stats = np.zeros((count, 5), dtype=np.int32)
    centroids = np.zeros((count, 2), dtype=np.float64)
    for label in range(1, count):
        ys, xs = np.nonzero(compact == label)
        if xs.size == 0:
            continue
        stats[label, cv2.CC_STAT_LEFT] = int(np.min(xs))
        stats[label, cv2.CC_STAT_TOP] = int(np.min(ys))
        stats[label, cv2.CC_STAT_WIDTH] = int(np.max(xs) - np.min(xs) + 1)
        stats[label, cv2.CC_STAT_HEIGHT] = int(np.max(ys) - np.min(ys) + 1)
        stats[label, cv2.CC_STAT_AREA] = int(xs.size)
        centroids[label] = (float(np.mean(xs)), float(np.mean(ys)))
    return count, compact, stats, centroids


def find_local_maxima(request: FindMaximaRequest) -> FindMaximaResult:
    source = _select_scalar_channel(request.image, request.channel).astype(np.float32)
    finite = np.isfinite(source)
    if not np.any(finite):
        return FindMaximaResult(
            maxima=(),
            candidate_plateau_count=0,
            suppressed_count=0,
            channel=request.channel,
            algorithm_version=request.algorithm_version,
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
    if request.algorithm_version == "1":
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
        resolved_prominence = (
            float(best_value - local_min[y, x])
            if request.algorithm_version == "1"
            else _topographic_prominence(
                safe,
                finite
                if request.roi_mask is None
                else finite & request.roi_mask,
                x=x,
                y=y,
                peak_value=best_value,
            )
        )
        if resolved_prominence + 1e-12 < request.prominence:
            continue
        plateau_maxima.append(
            LocalMaximum(
                x=float(x),
                y=float(y),
                value=best_value,
                local_prominence=resolved_prominence,
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
        algorithm_version=request.algorithm_version,
        request_id=request.request_id,
        generation=request.generation,
    )


def _topographic_prominence(
    values: NDArray[np.float32],
    allowed: NDArray[np.bool_],
    *,
    x: int,
    y: int,
    peak_value: float,
) -> float:
    """Return peak prominence from the highest saddle path to a higher peak."""

    import heapq

    height, width = values.shape
    best = np.full(values.shape, -np.inf, dtype=np.float32)
    best[y, x] = np.float32(peak_value)
    queue: list[tuple[float, int, int]] = [(-peak_value, y, x)]
    while queue:
        negative_capacity, current_y, current_x = heapq.heappop(queue)
        capacity = -negative_capacity
        if capacity + 1e-7 < float(best[current_y, current_x]):
            continue
        if (
            (current_x != x or current_y != y)
            and float(values[current_y, current_x]) > peak_value + 1e-7
        ):
            return max(0.0, float(peak_value - capacity))
        for offset_y, offset_x in (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ):
            neighbor_y = current_y + offset_y
            neighbor_x = current_x + offset_x
            if (
                neighbor_x < 0
                or neighbor_x >= width
                or neighbor_y < 0
                or neighbor_y >= height
                or not allowed[neighbor_y, neighbor_x]
            ):
                continue
            candidate = min(capacity, float(values[neighbor_y, neighbor_x]))
            if candidate <= float(best[neighbor_y, neighbor_x]) + 1e-7:
                continue
            best[neighbor_y, neighbor_x] = np.float32(candidate)
            heapq.heappush(queue, (-candidate, neighbor_y, neighbor_x))
    selected = values[allowed]
    base = float(np.min(selected)) if selected.size else peak_value
    return max(0.0, float(peak_value - base))


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


@dataclass(frozen=True, slots=True)
class _RingTopology:
    depths: tuple[int, ...]
    parents: tuple[int | None, ...]


def _classify_odd_even_rings(rings: ImmutableRings) -> _RingTopology:
    """Classify unordered, disjoint-or-nested rings by odd-even containment."""

    samples = tuple(_ring_interior_sample(ring) for ring in rings)
    magnitudes = tuple(abs(_signed_ring_area(ring)) for ring in rings)
    containers: list[tuple[int, ...]] = []
    for index, sample in enumerate(samples):
        containing = tuple(
            other_index
            for other_index, other in enumerate(rings)
            if other_index != index
            and magnitudes[other_index] > magnitudes[index] + 1e-9
            and cv2.pointPolygonTest(
                np.asarray(other, dtype=np.float32),
                sample,
                False,
            )
            > 0
        )
        containers.append(containing)
    depths = tuple(len(items) for items in containers)
    parents = tuple(
        (
            None
            if not items
            else min(items, key=lambda candidate: magnitudes[candidate])
        )
        for items in containers
    )
    return _RingTopology(depths=depths, parents=parents)


def _ring_interior_sample(ring: ImmutableRing) -> Coordinate:
    contour = np.asarray(ring, dtype=np.float64)
    signed_area = _signed_ring_area(ring)
    if abs(signed_area) > 1e-12:
        factor = 1.0 / (6.0 * signed_area)
        moment_x = 0.0
        moment_y = 0.0
        for index, (x0, y0) in enumerate(ring):
            x1, y1 = ring[(index + 1) % len(ring)]
            cross = (x0 * y1) - (x1 * y0)
            moment_x += (x0 + x1) * cross
            moment_y += (y0 + y1) * cross
        centroid = (moment_x * factor, moment_y * factor)
        if cv2.pointPolygonTest(
            contour.astype(np.float32),
            centroid,
            False,
        ) > 0:
            return centroid
    minimum = np.min(contour, axis=0)
    maximum = np.max(contour, axis=0)
    for division in (8, 16, 32):
        for row in range(division):
            y = float(
                minimum[1]
                + ((row + 0.5) / division) * (maximum[1] - minimum[1])
            )
            for column in range(division):
                x = float(
                    minimum[0]
                    + ((column + 0.5) / division) * (maximum[0] - minimum[0])
                )
                if cv2.pointPolygonTest(
                    contour.astype(np.float32),
                    (x, y),
                    False,
                ) > 0:
                    return x, y
    return float(contour[0, 0]), float(contour[0, 1])


def _shape_component_table(
    rings: ImmutableRings,
    topology: _RingTopology,
    *,
    pixel_size_x: float,
    pixel_size_y: float,
) -> tuple[ShapeComponentResult, ...]:
    area_scale = pixel_size_x * pixel_size_y
    outer_indices = tuple(
        index
        for index, depth in enumerate(topology.depths)
        if depth % 2 == 0
    )
    components: list[ShapeComponentResult] = []
    for component_index, outer_index in enumerate(outer_indices, start=1):
        outer_depth = topology.depths[outer_index]
        hole_indices = tuple(
            index
            for index, parent in enumerate(topology.parents)
            if parent == outer_index
            and topology.depths[index] == outer_depth + 1
            and topology.depths[index] % 2 == 1
        )
        component_rings = (rings[outer_index],) + tuple(
            rings[index] for index in hole_indices
        )
        point_rings = [
            [Point(float(x), float(y)) for x, y in ring]
            for ring in component_rings
        ]
        component_area_px, component_centroid = area_rings_area_and_centroid(
            point_rings
        )
        outer_perimeter_px = _ring_perimeter(rings[outer_index])
        hole_perimeter_px = sum(
            _ring_perimeter(rings[index]) for index in hole_indices
        )
        outer = rings[outer_index]
        xs = tuple(x for x, _y in outer)
        ys = tuple(y for _x, y in outer)
        bounds = (min(xs), min(ys), max(xs), max(ys))
        bounds_area = (
            (bounds[2] - bounds[0])
            * pixel_size_x
            * (bounds[3] - bounds[1])
            * pixel_size_y
        )
        physical_outer = np.asarray(
            [
                (x * pixel_size_x, y * pixel_size_y)
                for x, y in outer
            ],
            dtype=np.float64,
        )
        hull = _convex_hull(physical_outer)
        convex_area = (
            abs(_signed_ring_area(tuple(map(tuple, hull.tolist()))))
            if len(hull) >= 3
            else 0.0
        )
        component_area = float(component_area_px) * area_scale
        components.append(
            ShapeComponentResult(
                index=component_index,
                area_px=float(component_area_px),
                area=component_area,
                centroid_px=(
                    float(component_centroid.x),
                    float(component_centroid.y),
                ),
                centroid=(
                    float(component_centroid.x) * pixel_size_x,
                    float(component_centroid.y) * pixel_size_y,
                ),
                outer_perimeter_px=outer_perimeter_px,
                hole_perimeter_px=hole_perimeter_px,
                total_perimeter_px=outer_perimeter_px + hole_perimeter_px,
                hole_count=len(hole_indices),
                bounds_px=bounds,
                extent=component_area / bounds_area if bounds_area > 0 else None,
                convex_area=convex_area if convex_area > 0 else None,
                solidity=(
                    component_area / convex_area
                    if component_area > 0 and convex_area > 0
                    else None
                ),
            )
        )
    return tuple(components)


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


def _optional_finite(value: object, name: str) -> float | None:
    if value is None:
        return None
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


class ImageAnalysisService:
    """Stateless façade for worker-side analysis dispatch."""

    analyze_shape = staticmethod(analyze_shape)
    analyze_intensity = staticmethod(analyze_intensity)
    calculate_histogram = staticmethod(calculate_histogram)
    calculate_fft_power_spectrum = staticmethod(calculate_fft_power_spectrum)
    sample_intensity_profile = staticmethod(sample_intensity_profile)
    analyze_particles = staticmethod(analyze_particles)
    find_local_maxima = staticmethod(find_local_maxima)
