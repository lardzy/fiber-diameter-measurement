"""Deterministic, offscreen benchmarks for the measurement canvas.

The benchmark intentionally exercises ``DocumentCanvas.paintEvent`` through
``QWidget.render`` instead of timing isolated geometry helpers.  Scenario data
is generated from fixed formulas and a recorded seed, while timing and memory
figures are treated as observations rather than assertions.

Example::

    python -m fdm.canvas_benchmark --scenario areas_holes_300 --json
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from typing import Callable, Sequence

# This module is a developer tool and must also work on hosts without a display
# server.  ``setdefault`` preserves an explicit platform chosen by the caller.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import __version__ as pyside_version
from PySide6.QtCore import (
    QEvent,
    QEventLoop,
    QPoint,
    QPointF,
    Qt,
    qVersion,
)
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPicture
from PySide6.QtWidgets import QApplication

from fdm.area_display import area_derived_geometry_service
from fdm.geometry import Line, Point
from fdm.models import (
    Calibration,
    ImageDocument,
    ImageViewState,
    Measurement,
)
from fdm.settings import (
    AppSettings,
    MagicSegmentToolMode,
    MeasurementLabelStyleSettings,
)
from fdm.ui.canvas import DocumentCanvas
from fdm.ui.area_handle_cache import area_handle_display_cache
from fdm.ui.canvas_overlay_cache import (
    CanvasOverlayCacheStats,
    CanvasOverlayTileKey,
    canvas_overlay_tile_cache,
)
from fdm.ui.digital_slide_canvas import DigitalSlideCanvas
from fdm.ui.screen_label_sprite_cache import screen_label_sprite_cache
from fdm.ui.view_transform import MAX_VIEW_ZOOM, MIN_VIEW_ZOOM
from fdm.services.digital_slide_store import (
    DIGITAL_SLIDE_TILE_CODEC_JPEG,
    DIGITAL_SLIDE_TILE_CODEC_PNG,
    DigitalSlideManifest,
    DigitalSlideStore,
    DigitalSlideTile,
)
from fdm.version import __version__


SCHEMA_VERSION = 1
DEFAULT_SEED = 20260719
DEFAULT_CANVAS_SIZE = (1024, 768)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_OUTPUT_ROOT = PROJECT_ROOT / ".tmp" / "canvas-benchmark"
_PAINTER_TRACE_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class ScenarioDefinition:
    name: str
    family: str
    default_object_count: int
    labels_enabled: bool
    description: str
    default_coordinate_count: int | None = None
    composition: str = "spaced"


@dataclass(slots=True)
class ScenarioData:
    definition: ScenarioDefinition
    document: ImageDocument
    image: QImage
    settings: AppSettings
    object_count: int
    coordinate_count: int
    canvas_size: tuple[int, int]
    seed: int


SCENARIOS: dict[str, ScenarioDefinition] = {
    definition.name: definition
    for definition in (
        ScenarioDefinition(
            name="length_labels_500",
            family="length",
            default_object_count=500,
            labels_enabled=True,
            description="500 visible straight-length measurements with labels.",
        ),
        ScenarioDefinition(
            name="length_labels_1000",
            family="length",
            default_object_count=1000,
            labels_enabled=True,
            description="1,000 visible straight-length measurements with labels.",
        ),
        ScenarioDefinition(
            name="length_no_labels_500",
            family="length",
            default_object_count=500,
            labels_enabled=False,
            description="500 visible straight-length measurements without labels.",
        ),
        ScenarioDefinition(
            name="length_no_labels_1000",
            family="length",
            default_object_count=1000,
            labels_enabled=False,
            description="1,000 visible straight-length measurements without labels.",
        ),
        ScenarioDefinition(
            name="areas_holes_100",
            family="area",
            default_object_count=100,
            labels_enabled=True,
            description="100 visible area measurements, each containing a hole.",
        ),
        ScenarioDefinition(
            name="areas_holes_300",
            family="area",
            default_object_count=300,
            labels_enabled=True,
            description="300 visible area measurements, each containing a hole.",
        ),
        ScenarioDefinition(
            name="areas_holes_500",
            family="area",
            default_object_count=500,
            labels_enabled=True,
            description="500 visible area measurements, each containing a hole.",
        ),
        ScenarioDefinition(
            name="area_coordinates_200000",
            family="area",
            default_object_count=100,
            labels_enabled=True,
            default_coordinate_count=200_000,
            description="Visible areas with 200,000 total RAW ring coordinates.",
        ),
        ScenarioDefinition(
            name="area_coordinates_600000",
            family="area",
            default_object_count=100,
            labels_enabled=True,
            default_coordinate_count=600_000,
            description="Visible areas with 600,000 total RAW ring coordinates.",
        ),
        ScenarioDefinition(
            name="magic_wand_dense_110",
            family="magic_area",
            default_object_count=110,
            labels_enabled=True,
            description=(
                "110 dense standard-magic-wand areas built from real "
                "mask-to-rings geometry."
            ),
            composition="dense",
        ),
        ScenarioDefinition(
            name="magic_wand_overlap_110",
            family="magic_area",
            default_object_count=110,
            labels_enabled=True,
            description=(
                "110 overlapping standard-magic-wand areas exercising "
                "composition-sensitive overlay tiles."
            ),
            composition="overlap",
        ),
        ScenarioDefinition(
            name="offscreen_5000",
            family="offscreen",
            default_object_count=5_000,
            labels_enabled=True,
            description="5,000 straight-length measurements outside the viewport.",
        ),
    )
}


class _BenchmarkCanvas(DocumentCanvas):
    """Canvas variant that records how many paint events the run requested."""

    def __init__(self, *, device_pixel_ratio: float = 1.0) -> None:
        self._benchmark_device_pixel_ratio = max(
            1.0,
            float(device_pixel_ratio),
        )
        super().__init__()
        self.paint_event_count = 0

    def devicePixelRatioF(self) -> float:  # noqa: N802 - Qt virtual name
        return float(self._benchmark_device_pixel_ratio)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt virtual name
        self.paint_event_count += 1
        super().paintEvent(event)


class _BenchmarkDigitalSlideCanvas(DigitalSlideCanvas):
    """Digital-slide canvas that records real renderer submissions."""

    def __init__(self, *, device_pixel_ratio: float = 1.0) -> None:
        self._benchmark_device_pixel_ratio = max(
            1.0,
            float(device_pixel_ratio),
        )
        super().__init__()
        self.paint_event_count = 0
        self.viewport_buffer_request_count = 0

    def devicePixelRatioF(self) -> float:  # noqa: N802 - Qt virtual name
        return float(self._benchmark_device_pixel_ratio)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt virtual name
        self.paint_event_count += 1
        super().paintEvent(event)

    def _request_display_frame(self) -> None:
        self.viewport_buffer_request_count += 1
        super()._request_display_frame()

    def _request_native_frame(self) -> None:
        self.viewport_buffer_request_count += 1
        super()._request_native_frame()

    def _request_coarse_frame(self) -> None:
        self.viewport_buffer_request_count += 1
        super()._request_coarse_frame()


class _BenchmarkDigitalSlideStore:
    """Owned temporary SQLite/PNG/JPEG fixture for renderer benchmarks."""

    def __init__(
        self,
        *,
        manifest: DigitalSlideManifest,
        fill_color: QColor,
    ) -> None:
        self._temporary = tempfile.TemporaryDirectory(
            prefix="fdm-canvas-benchmark-slide-"
        )
        self.path = Path(self._temporary.name) / "benchmark.fdmslide"
        self._manifest = manifest
        store = DigitalSlideStore.create(self.path, manifest)
        tile_width = max(1, int(manifest.viewport_width))
        tile_height = max(1, int(manifest.viewport_height))
        tile_index = 0
        for y in range(0, int(manifest.height), tile_height):
            for x in range(0, int(manifest.width), tile_width):
                width = min(tile_width, int(manifest.width) - x)
                height = min(tile_height, int(manifest.height) - y)
                image = QImage(width, height, QImage.Format.Format_RGB32)
                color = QColor(fill_color)
                color.setRed((color.red() + x // max(1, tile_width) * 23) % 256)
                color.setGreen((color.green() + y // max(1, tile_height) * 29) % 256)
                image.fill(color)
                codec = (
                    DIGITAL_SLIDE_TILE_CODEC_PNG
                    if tile_index % 2 == 0
                    else DIGITAL_SLIDE_TILE_CODEC_JPEG
                )
                store.write_tile(
                    DigitalSlideTile(
                        z_index=0,
                        x=x,
                        y=y,
                        width=width,
                        height=height,
                    ),
                    image,
                    codec=codec,
                    quality=90,
                    update_manifest=False,
                )
                tile_index += 1
        manifest.tile_count = tile_index
        store.write_manifest(manifest)
        store.close()

    def read_manifest(self) -> DigitalSlideManifest:
        return self._manifest

    def close(self) -> None:
        self._temporary.cleanup()


def _label_settings(enabled: bool) -> AppSettings:
    font_family = {
        "Darwin": "PingFang SC",
        "Windows": "Microsoft YaHei UI",
    }.get(platform.system(), "DejaVu Sans")
    style = MeasurementLabelStyleSettings(
        enabled=enabled,
        font_family=font_family,
        font_size=14,
        color="#F4F1DE",
        # Three decimals keep the deterministic 1,000-object length workload
        # from collapsing adjacent values onto the same formatted label.
        decimals=3,
        background_enabled=True,
        parallel_to_line=False,
    )
    return AppSettings(
        length_measurement_label_style=replace(style),
        area_measurement_label_style=replace(style),
    )


def _fit_view_state(
    image_size: tuple[int, int],
    canvas_size: tuple[int, int],
    *,
    margin: float = 24.0,
) -> ImageViewState:
    image_width, image_height = image_size
    canvas_width, canvas_height = canvas_size
    zoom = min(
        max(1.0, canvas_width - (margin * 2.0)) / image_width,
        max(1.0, canvas_height - (margin * 2.0)) / image_height,
    )
    pan = Point(
        (canvas_width - (image_width * zoom)) / 2.0,
        (canvas_height - (image_height * zoom)) / 2.0,
    )
    return ImageViewState(zoom=zoom, pan=pan)


def _background_image(image_size: tuple[int, int]) -> QImage:
    image = QImage(
        image_size[0],
        image_size[1],
        QImage.Format.Format_RGB32,
    )
    image.fill(QColor("#25313C"))
    return image


def _benchmark_document(
    *,
    scenario_name: str,
    image_size: tuple[int, int],
    measurements: list[Measurement],
    view_state: ImageViewState,
) -> ImageDocument:
    return ImageDocument(
        id=f"benchmark-{scenario_name}",
        path="",
        image_size=image_size,
        source_type="benchmark",
        calibration=Calibration(
            mode="benchmark",
            pixels_per_unit=2.0,
            unit="μm",
            source_label="deterministic canvas benchmark",
        ),
        measurements=measurements,
        view_state=view_state,
    )


def _visible_line_measurements(
    count: int,
    *,
    scenario_name: str,
    image_size: tuple[int, int],
    seed: int,
) -> list[Measurement]:
    width, height = image_size
    aspect = width / max(1.0, float(height))
    columns = max(1, math.ceil(math.sqrt(count * aspect)))
    rows = max(1, math.ceil(count / columns))
    cell_width = (width - 80.0) / columns
    cell_height = (height - 80.0) / rows
    measurements: list[Measurement] = []
    phase = (seed % 360) * math.pi / 180.0
    for index in range(count):
        column = index % columns
        row = index // columns
        center_x = 40.0 + ((column + 0.5) * cell_width)
        center_y = 40.0 + ((row + 0.5) * cell_height)
        angle = phase + ((index * 0.61803398875) % 1.0) * math.pi
        # Keep every result label distinct at the configured two decimals.
        # Reusing one nominal length would benchmark a single sprite-cache
        # entry even in the advertised 500/1,000-label scenarios.
        length_scale = 0.72 + (0.26 * ((index + 1) / (count + 1)))
        half_length = max(
            3.0,
            min(cell_width, cell_height) * 0.33 * length_scale,
        )
        dx = math.cos(angle) * half_length
        dy = math.sin(angle) * half_length
        measurements.append(
            Measurement(
                id=f"{scenario_name}-line-{index:06d}",
                image_id=f"benchmark-{scenario_name}",
                fiber_group_id=None,
                mode="manual",
                measurement_kind="line",
                line_px=Line(
                    Point(center_x - dx, center_y - dy),
                    Point(center_x + dx, center_y + dy),
                ),
                diameter_px=half_length * 2.0,
                diameter_unit=half_length,
            )
        )
    return measurements


def _offscreen_line_measurements(
    count: int,
    *,
    scenario_name: str,
    image_size: tuple[int, int],
    canvas_size: tuple[int, int],
) -> tuple[list[Measurement], ImageViewState]:
    width, height = image_size
    start_x = min(float(width) - 40.0, float(canvas_size[0]) + 180.0)
    usable_width = max(20.0, float(width) - start_x - 20.0)
    usable_height = max(20.0, float(height) - 40.0)
    columns = max(1, math.ceil(math.sqrt(count * usable_width / usable_height)))
    rows = max(1, math.ceil(count / columns))
    cell_width = usable_width / columns
    cell_height = usable_height / rows
    measurements: list[Measurement] = []
    for index in range(count):
        column = index % columns
        row = index // columns
        x = start_x + ((column + 0.25) * cell_width)
        y = 20.0 + ((row + 0.5) * cell_height)
        line_length = max(1.0, min(8.0, cell_width * 0.4))
        measurements.append(
            Measurement(
                id=f"{scenario_name}-line-{index:06d}",
                image_id=f"benchmark-{scenario_name}",
                fiber_group_id=None,
                mode="manual",
                measurement_kind="line",
                line_px=Line(Point(x, y), Point(x + line_length, y + 1.0)),
                diameter_px=line_length,
                diameter_unit=line_length / 2.0,
            )
        )
    # Both set_document() and the first resize event can fit an exact 1.0 zoom
    # to the whole image. Keep an imperceptibly different zoom so the benchmark
    # really leaves every generated object outside the viewport.
    return measurements, ImageViewState(
        zoom=1.000001,
        pan=Point(0.01, 0.01),
    )


def _ring(
    *,
    center_x: float,
    center_y: float,
    radius_x: float,
    radius_y: float,
    vertices: int,
    phase: float,
    clockwise: bool,
) -> list[Point]:
    direction = -1.0 if clockwise else 1.0
    points: list[Point] = []
    for index in range(vertices):
        angle = phase + direction * (math.tau * index / vertices)
        ripple = 1.0 + (0.045 * math.sin((index * 7.0) + phase))
        points.append(
            Point(
                center_x + (math.cos(angle) * radius_x * ripple),
                center_y + (math.sin(angle) * radius_y * ripple),
            )
        )
    return points


def _coordinates_per_area(
    *,
    object_count: int,
    requested_total: int | None,
) -> list[int]:
    if requested_total is None:
        return [96] * object_count
    minimum_total = object_count * 8
    if requested_total < minimum_total:
        raise ValueError(
            f"area coordinate count must be at least {minimum_total} "
            f"for {object_count} objects"
        )
    base, remainder = divmod(requested_total, object_count)
    return [base + (1 if index < remainder else 0) for index in range(object_count)]


def _area_measurements(
    count: int,
    *,
    scenario_name: str,
    image_size: tuple[int, int],
    seed: int,
    requested_coordinate_count: int | None,
) -> list[Measurement]:
    width, height = image_size
    aspect = width / max(1.0, float(height))
    columns = max(1, math.ceil(math.sqrt(count * aspect)))
    rows = max(1, math.ceil(count / columns))
    cell_width = (width - 48.0) / columns
    cell_height = (height - 48.0) / rows
    coordinates_per_area = _coordinates_per_area(
        object_count=count,
        requested_total=requested_coordinate_count,
    )
    measurements: list[Measurement] = []
    seed_phase = (seed % 997) / 997.0 * math.tau
    for index, total_vertices in enumerate(coordinates_per_area):
        column = index % columns
        row = index // columns
        center_x = 24.0 + ((column + 0.5) * cell_width)
        center_y = 24.0 + ((row + 0.5) * cell_height)
        outer_vertices = max(4, int(round(total_vertices * 0.7)))
        hole_vertices = total_vertices - outer_vertices
        if hole_vertices < 4:
            hole_vertices = 4
            outer_vertices = total_vertices - hole_vertices
        phase = seed_phase + ((index * 0.38196601125) % 1.0) * math.tau
        size_scale = 0.78 + (0.20 * ((index + 1) / (count + 1)))
        outer_radius_x = max(2.0, cell_width * 0.36 * size_scale)
        outer_radius_y = max(2.0, cell_height * 0.36 * size_scale)
        hole_radius_x = max(0.7, cell_width * 0.12 * size_scale)
        hole_radius_y = max(0.7, cell_height * 0.12 * size_scale)
        outer = _ring(
            center_x=center_x,
            center_y=center_y,
            radius_x=outer_radius_x,
            radius_y=outer_radius_y,
            vertices=outer_vertices,
            phase=phase,
            clockwise=False,
        )
        hole = _ring(
            center_x=center_x + (cell_width * 0.025),
            center_y=center_y - (cell_height * 0.02),
            radius_x=hole_radius_x,
            radius_y=hole_radius_y,
            vertices=hole_vertices,
            phase=phase * 0.73,
            clockwise=True,
        )
        nominal_area = max(
            1.0,
            math.pi
            * (
                (outer_radius_x * outer_radius_y)
                - (hole_radius_x * hole_radius_y)
            ),
        )
        measurements.append(
            Measurement(
                id=f"{scenario_name}-area-{index:06d}",
                image_id=f"benchmark-{scenario_name}",
                fiber_group_id=None,
                mode="area",
                measurement_kind="area",
                polygon_px=outer,
                area_rings_px=[outer, hole],
                exact_area_px=nominal_area,
                area_px=nominal_area,
                area_unit=nominal_area / 4.0,
            )
        )
    return measurements


_MAGIC_WAND_TEMPLATE: tuple[
    tuple[tuple[tuple[float, float], ...], ...],
    tuple[tuple[float, float], ...],
    float,
] | None = None


def _magic_wand_geometry_template() -> tuple[
    tuple[tuple[tuple[float, float], ...], ...],
    tuple[tuple[float, float], ...],
    float,
]:
    """Build one deterministic standard-wand contour through production code.

    A smooth synthetic ellipse substantially understates the cost of a real
    segmentation boundary.  This binary fixture combines several boundary
    frequencies and one hole, then uses the same ``magic_mask_to_geometry``
    conversion as a committed standard-wand measurement.  Coordinate tuples
    are cached instead of mutable ``Point`` objects so scenarios never share
    editable geometry.
    """

    global _MAGIC_WAND_TEMPLATE
    if _MAGIC_WAND_TEMPLATE is not None:
        return _MAGIC_WAND_TEMPLATE

    import cv2
    import numpy as np

    from fdm.services.prompt_segmentation import (
        magic_mask_area_px,
        magic_mask_to_geometry,
    )

    sample_count = 1_440
    angles = np.linspace(0.0, math.tau, sample_count, endpoint=False)
    radius = (
        180.0
        + (18.0 * np.sin(13.0 * angles))
        + (8.0 * np.sin(37.0 * angles))
        + (4.0 * np.sin(83.0 * angles))
    )
    contour = np.stack(
        (
            256.0 + (radius * np.cos(angles)),
            256.0 + (radius * np.sin(angles)),
        ),
        axis=1,
    ).round().astype(np.int32)
    mask = np.zeros((512, 512), dtype=np.uint8)
    cv2.fillPoly(mask, [contour], 1)
    cv2.circle(mask, (280, 244), 38, 0, thickness=cv2.FILLED)
    selected_mask, rings, polygon, _stats = magic_mask_to_geometry(
        mask.astype(bool),
        select_prompt_component=False,
    )
    if selected_mask is None or not rings or len(polygon) < 3:
        raise RuntimeError("deterministic magic-wand mask produced no geometry")

    all_points = [point for ring in rings for point in ring]
    center_x = (
        min(point.x for point in all_points)
        + max(point.x for point in all_points)
    ) / 2.0
    center_y = (
        min(point.y for point in all_points)
        + max(point.y for point in all_points)
    ) / 2.0
    normalized_rings = tuple(
        tuple(
            (float(point.x) - center_x, float(point.y) - center_y)
            for point in ring
        )
        for ring in rings
    )
    normalized_polygon = tuple(
        (float(point.x) - center_x, float(point.y) - center_y)
        for point in polygon
    )
    _MAGIC_WAND_TEMPLATE = (
        normalized_rings,
        normalized_polygon,
        float(magic_mask_area_px(selected_mask)),
    )
    return _MAGIC_WAND_TEMPLATE


def _resample_closed_coordinates(
    coordinates: Sequence[tuple[float, float]],
    target_count: int,
) -> list[tuple[float, float]]:
    """Resample one closed ring to an exact count for small test overrides."""

    count = max(3, int(target_count))
    if count == len(coordinates):
        return list(coordinates)
    if not coordinates:
        return []
    segment_lengths: list[float] = []
    perimeter = 0.0
    for index, start in enumerate(coordinates):
        end = coordinates[(index + 1) % len(coordinates)]
        length = math.hypot(end[0] - start[0], end[1] - start[1])
        segment_lengths.append(length)
        perimeter += length
    if perimeter <= 1e-9:
        return [coordinates[index % len(coordinates)] for index in range(count)]

    result: list[tuple[float, float]] = []
    segment_index = 0
    segment_start_distance = 0.0
    for sample_index in range(count):
        target_distance = perimeter * sample_index / count
        while (
            segment_index < len(segment_lengths) - 1
            and target_distance
            > segment_start_distance + segment_lengths[segment_index]
        ):
            segment_start_distance += segment_lengths[segment_index]
            segment_index += 1
        start = coordinates[segment_index]
        end = coordinates[(segment_index + 1) % len(coordinates)]
        length = segment_lengths[segment_index]
        ratio = (
            0.0
            if length <= 1e-9
            else (target_distance - segment_start_distance) / length
        )
        result.append(
            (
                start[0] + ((end[0] - start[0]) * ratio),
                start[1] + ((end[1] - start[1]) * ratio),
            )
        )
    return result


def _magic_wand_measurements(
    count: int,
    *,
    scenario_name: str,
    image_size: tuple[int, int],
    seed: int,
    requested_coordinate_count: int | None,
    composition: str,
) -> list[Measurement]:
    """Create dense or overlapping measurements from real mask-derived rings."""

    template_rings, template_polygon, template_area = (
        _magic_wand_geometry_template()
    )
    width, height = image_size
    aspect = width / max(1.0, float(height))
    columns = max(1, math.ceil(math.sqrt(count * aspect)))
    rows = max(1, math.ceil(count / columns))
    cell_width = (width - 64.0) / columns
    cell_height = (height - 64.0) / rows
    template_points = [
        point
        for ring in template_rings
        for point in ring
    ]
    half_width = max(abs(point[0]) for point in template_points)
    half_height = max(abs(point[1]) for point in template_points)
    radius_fraction = 0.58 if composition == "overlap" else 0.38
    per_object_counts = (
        _coordinates_per_area(
            object_count=count,
            requested_total=requested_coordinate_count,
        )
        if requested_coordinate_count is not None
        else None
    )
    original_total = sum(len(ring) for ring in template_rings)
    outer_fraction = len(template_rings[0]) / max(1, original_total)
    measurements: list[Measurement] = []
    seed_phase = (seed % 997) / 997.0 * math.tau

    for index in range(count):
        column = index % columns
        row = index // columns
        center_x = 32.0 + ((column + 0.5) * cell_width)
        center_y = 32.0 + ((row + 0.5) * cell_height)
        if composition == "dense":
            center_x += (((index * 37) % 11) - 5) * cell_width * 0.008
            center_y += (((index * 53) % 11) - 5) * cell_height * 0.008

        if per_object_counts is None:
            source_rings = [list(ring) for ring in template_rings]
        else:
            total = per_object_counts[index]
            outer_count = max(4, int(round(total * outer_fraction)))
            hole_count = max(4, total - outer_count)
            outer_count = total - hole_count
            source_rings = [
                _resample_closed_coordinates(template_rings[0], outer_count),
                _resample_closed_coordinates(template_rings[1], hole_count),
            ]

        size_scale = 0.92 + (0.12 * ((index % 17) / 16.0))
        scale_x = (
            (cell_width * radius_fraction) / max(half_width, 1e-9)
        ) * size_scale
        scale_y = (
            (cell_height * radius_fraction) / max(half_height, 1e-9)
        ) * size_scale
        angle = seed_phase + ((index % 9) - 4) * 0.025
        cosine = math.cos(angle)
        sine = math.sin(angle)

        def transform(coordinate: tuple[float, float]) -> Point:
            x = coordinate[0] * scale_x
            y = coordinate[1] * scale_y
            return Point(
                center_x + (x * cosine) - (y * sine),
                center_y + (x * sine) + (y * cosine),
            )

        rings = [
            [transform(coordinate) for coordinate in ring]
            for ring in source_rings
        ]
        polygon = [
            transform(coordinate)
            for coordinate in template_polygon
        ]
        exact_area = max(1.0, template_area * scale_x * scale_y)
        measurements.append(
            Measurement(
                id=f"{scenario_name}-magic-{index:06d}",
                image_id=f"benchmark-{scenario_name}",
                fiber_group_id=None,
                mode="magic_segment",
                measurement_kind="area",
                polygon_px=polygon,
                area_rings_px=rings,
                exact_area_px=exact_area,
                area_px=exact_area,
                area_unit=exact_area / 4.0,
                confidence=1.0,
                status="manual",
                debug_payload={
                    "benchmark_geometry_source": "magic_mask_to_geometry",
                    "benchmark_composition": composition,
                },
            )
        )
    return measurements


def build_scenario(
    scenario_name: str,
    *,
    object_count: int | None = None,
    coordinate_count: int | None = None,
    canvas_size: tuple[int, int] = DEFAULT_CANVAS_SIZE,
    seed: int = DEFAULT_SEED,
) -> ScenarioData:
    """Build one deterministic scenario without displaying a window."""

    try:
        definition = SCENARIOS[scenario_name]
    except KeyError as exc:
        available = ", ".join(sorted(SCENARIOS))
        raise ValueError(f"unknown scenario {scenario_name!r}; choose from {available}") from exc
    resolved_count = (
        definition.default_object_count if object_count is None else int(object_count)
    )
    if resolved_count <= 0:
        raise ValueError("object_count must be greater than zero")
    if canvas_size[0] <= 0 or canvas_size[1] <= 0:
        raise ValueError("canvas dimensions must be greater than zero")

    if definition.family == "offscreen":
        image_size = (
            max(2048, canvas_size[0] + 1024),
            max(1536, canvas_size[1] + 512),
        )
        measurements, view_state = _offscreen_line_measurements(
            resolved_count,
            scenario_name=scenario_name,
            image_size=image_size,
            canvas_size=canvas_size,
        )
    elif definition.family == "length":
        if coordinate_count is not None:
            raise ValueError("--coordinates is only valid for area scenarios")
        image_size = (1600, 1200)
        measurements = _visible_line_measurements(
            resolved_count,
            scenario_name=scenario_name,
            image_size=image_size,
            seed=seed,
        )
        view_state = _fit_view_state(image_size, canvas_size)
    elif definition.family == "magic_area":
        image_size = (2400, 1600)
        measurements = _magic_wand_measurements(
            resolved_count,
            scenario_name=scenario_name,
            image_size=image_size,
            seed=seed,
            requested_coordinate_count=coordinate_count,
            composition=definition.composition,
        )
        view_state = _fit_view_state(image_size, canvas_size)
    else:
        image_size = (1800, 1200)
        requested_coordinates = (
            definition.default_coordinate_count
            if coordinate_count is None
            else int(coordinate_count)
        )
        measurements = _area_measurements(
            resolved_count,
            scenario_name=scenario_name,
            image_size=image_size,
            seed=seed,
            requested_coordinate_count=requested_coordinates,
        )
        view_state = _fit_view_state(image_size, canvas_size)

    actual_coordinate_count = sum(
        (
            len(measurement.area_rings_px[0])
            + sum(len(ring) for ring in measurement.area_rings_px[1:])
        )
        if measurement.measurement_kind == "area"
        else 2
        for measurement in measurements
    )
    document = _benchmark_document(
        scenario_name=scenario_name,
        image_size=image_size,
        measurements=measurements,
        view_state=view_state,
    )
    if definition.family == "magic_area" and document.measurements:
        # Standard-wand commits select the newest area, which remains an exact
        # active layer above the passive tile cache during navigation.
        document.select_measurement(document.measurements[-1].id)
    return ScenarioData(
        definition=definition,
        document=document,
        image=_background_image(image_size),
        settings=_label_settings(definition.labels_enabled),
        object_count=len(measurements),
        coordinate_count=actual_coordinate_count,
        canvas_size=canvas_size,
        seed=seed,
    )


def _percentile(samples: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(sample) for sample in samples)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] + ((ordered[upper] - ordered[lower]) * weight)


def _timing_summary(samples: Sequence[float]) -> dict[str, object]:
    rounded_samples = [round(float(sample), 6) for sample in samples]
    return {
        "frame_count": len(samples),
        "p50": round(_percentile(samples, 0.50), 6),
        "p95": round(_percentile(samples, 0.95), 6),
        "max": round(max(samples, default=0.0), 6),
        "mean": round(statistics.fmean(samples) if samples else 0.0, 6),
        "samples": rounded_samples,
    }


def _current_rss_bytes() -> tuple[int | None, str | None]:
    try:
        import psutil  # type: ignore[import-not-found]

        return int(psutil.Process().memory_info().rss), "psutil"
    except (ImportError, OSError):
        pass

    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            process = ctypes.windll.kernel32.GetCurrentProcess()
            succeeded = ctypes.windll.psapi.GetProcessMemoryInfo(
                process,
                ctypes.byref(counters),
                counters.cb,
            )
            if succeeded:
                return int(counters.WorkingSetSize), "win32"
        except (AttributeError, OSError, TypeError, ValueError):
            pass

    try:
        completed = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            return int(completed.stdout.strip().split()[0]) * 1024, "ps"
    except (FileNotFoundError, OSError, subprocess.SubprocessError, ValueError):
        pass
    return None, None


def _peak_rss_bytes() -> tuple[int | None, str | None]:
    try:
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform != "darwin":
            peak *= 1024
        return peak, "resource"
    except (ImportError, OSError, ValueError):
        return None, None


def _git_environment() -> dict[str, str | None]:
    def run_git(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=PROJECT_ROOT,
                check=False,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
        except (FileNotFoundError, OSError, subprocess.SubprocessError):
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout.strip() or None

    return {
        "commit": run_git("rev-parse", "HEAD"),
        "branch": run_git("branch", "--show-current"),
    }


def _environment_payload(app: QApplication, *, canvas: DocumentCanvas) -> dict[str, object]:
    return {
        "fdm_version": __version__,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "pyside_version": pyside_version,
        "qt_version": qVersion(),
        "qt_platform": app.platformName(),
        "qt_qpa_platform_env": os.environ.get("QT_QPA_PLATFORM"),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "logical_cpu_count": os.cpu_count(),
        "device_pixel_ratio": round(float(canvas.devicePixelRatioF()), 4),
        "git": _git_environment(),
    }


@dataclass(slots=True)
class _PainterCallTrace:
    draw_path: int = 0
    draw_image: int = 0
    draw_pixmap: int = 0
    picture_play: int = 0


@contextmanager
def _trace_painter_calls(canvas: DocumentCanvas):
    """Count explicit UI-thread QPainter path/image calls for one trace frame.

    The production renderer constructs its own ``QPainter`` inside
    ``paintEvent``.  An unmeasured trace frame is therefore the only way to
    count those calls without changing production call sites.  Restricting the
    wrappers to the canvas' Qt thread excludes any concurrent overlay worker
    painters, and timing samples are always collected outside this context.
    """

    trace = _PainterCallTrace()
    owner_thread_id = threading.get_ident()
    original_draw_path = QPainter.drawPath
    original_draw_image = QPainter.drawImage
    original_draw_pixmap = QPainter.drawPixmap
    original_picture_play = QPicture.play

    def traced_draw_path(painter, *args, **kwargs):
        if threading.get_ident() == owner_thread_id:
            trace.draw_path += 1
        return original_draw_path(painter, *args, **kwargs)

    def traced_draw_image(painter, *args, **kwargs):
        if threading.get_ident() == owner_thread_id:
            trace.draw_image += 1
        return original_draw_image(painter, *args, **kwargs)

    def traced_draw_pixmap(painter, *args, **kwargs):
        if threading.get_ident() == owner_thread_id:
            trace.draw_pixmap += 1
        return original_draw_pixmap(painter, *args, **kwargs)

    def traced_picture_play(picture, *args, **kwargs):
        if threading.get_ident() == owner_thread_id:
            trace.picture_play += 1
        return original_picture_play(picture, *args, **kwargs)

    with _PAINTER_TRACE_LOCK:
        QPainter.drawPath = traced_draw_path
        QPainter.drawImage = traced_draw_image
        QPainter.drawPixmap = traced_draw_pixmap
        QPicture.play = traced_picture_play
        try:
            yield trace
        finally:
            QPainter.drawPath = original_draw_path
            QPainter.drawImage = original_draw_image
            QPainter.drawPixmap = original_draw_pixmap
            QPicture.play = original_picture_play


def _render_call_trace(
    canvas: DocumentCanvas,
    surface: QImage,
) -> dict[str, object]:
    with _trace_painter_calls(canvas) as trace:
        _render_frame(canvas, surface)
    return {
        "scope": "one unmeasured canvas render on the UI thread",
        "timed": False,
        "render_count": 1,
        "draw_path": int(trace.draw_path),
        "draw_image": int(trace.draw_image),
        "draw_pixmap": int(trace.draw_pixmap),
        "picture_play": int(trace.picture_play),
    }


def _runtime_cache_snapshot() -> dict[str, object]:
    label_stats = screen_label_sprite_cache.stats()
    handle_stats = area_handle_display_cache.stats()
    tile_stats = canvas_overlay_tile_cache.stats()
    return {
        "area_paths": {
            "entries": int(area_derived_geometry_service.path_cache_entry_count),
            "bytes": int(area_derived_geometry_service.path_cache_bytes),
            "generation": int(area_derived_geometry_service.path_cache_generation),
        },
        "label_sprites": {
            "entries": int(label_stats.entries),
            "bytes": int(label_stats.bytes),
            "max_bytes": int(label_stats.max_bytes),
            "hits": int(label_stats.hits),
            "misses": int(label_stats.misses),
            "evictions": int(label_stats.evictions),
        },
        "area_handles": {
            "entries": int(handle_stats.entries),
            "bytes": int(handle_stats.bytes),
            "hits": int(handle_stats.hits),
            "misses": int(handle_stats.misses),
            "evictions": int(handle_stats.evictions),
        },
        "overlay_tiles": {
            "entries": int(tile_stats.entries),
            "bytes": int(tile_stats.bytes),
            "pending": int(tile_stats.pending),
            "pending_bytes": int(tile_stats.pending_bytes),
        },
    }


def _cache_counter_delta(
    before: dict[str, object],
    after: dict[str, object],
    *,
    cache_name: str,
    fields: Sequence[str],
) -> dict[str, int]:
    before_cache = before[cache_name]
    after_cache = after[cache_name]
    assert isinstance(before_cache, dict)
    assert isinstance(after_cache, dict)
    return {
        field: max(
            0,
            int(after_cache.get(field, 0)) - int(before_cache.get(field, 0)),
        )
        for field in fields
    }


def _reset_runtime_caches(
    app: QApplication,
    *,
    drain_timeout_ms: int = 2_000,
) -> dict[str, object]:
    """Start each benchmark with process-global display caches genuinely cold."""

    canvas_overlay_tile_cache.clear()
    started = time.perf_counter()
    deadline = started + (max(0, int(drain_timeout_ms)) / 1000.0)
    while (
        canvas_overlay_tile_cache.stats().pending_bytes > 0
        and time.perf_counter() < deadline
    ):
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 10)
        time.sleep(0.001)
    app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 10)
    preexisting_pending_bytes = int(
        canvas_overlay_tile_cache.stats().pending_bytes
    )
    # Drain can publish cancelled completions into the aggregate counters. A
    # second clear removes any payload while the following baseline captures
    # the new run's own counter values.
    canvas_overlay_tile_cache.clear()
    area_derived_geometry_service.clear()
    area_handle_display_cache.clear()
    screen_label_sprite_cache.clear(reset_stats=True)
    return {
        "drain_timeout_ms": int(drain_timeout_ms),
        "drained": preexisting_pending_bytes == 0,
        "pending_bytes_after_drain": preexisting_pending_bytes,
    }


@dataclass(slots=True)
class _OverlayDropReasonTrace:
    generation_late: int = 0
    other_defensive: int = 0


@contextmanager
def _trace_overlay_drop_reasons(canvas: DocumentCanvas):
    """Split stale-generation completions from other cache rejections.

    ``CanvasOverlayTileCache`` deliberately keeps one aggregate defensive drop
    counter.  The benchmark observes its three completion admission points and
    classifies a drop as generation-late only when the canvas rejects the
    completed key as no longer current.  This leaves the production cache API
    untouched while giving performance dashboards an unambiguous late-result
    metric.
    """

    trace = _OverlayDropReasonTrace()
    cache = canvas_overlay_tile_cache
    method_names = ("_drop_completion", "_on_completed", "_on_failed")
    originals = {name: getattr(cache, name) for name in method_names}

    def classify(
        original: Callable[..., object],
        key: CanvasOverlayTileKey,
        *args,
    ) -> object:
        dropped_before = int(cache.stats().dropped)
        stale_generation = not canvas._overlay_tile_key_is_current(key)
        result = original(key, *args)
        dropped_after = int(cache.stats().dropped)
        delta = max(0, dropped_after - dropped_before)
        if stale_generation:
            trace.generation_late += delta
        else:
            trace.other_defensive += delta
        return result

    for name, original in originals.items():
        setattr(
            cache,
            name,
            lambda key, *args, _original=original: classify(
                _original,
                key,
                *args,
            ),
        )
    try:
        yield trace
    finally:
        for name in method_names:
            delattr(cache, name)


def _render_frame(canvas: DocumentCanvas, surface: QImage) -> float:
    surface.fill(QColor(0, 0, 0, 0))
    started = time.perf_counter_ns()
    painter = QPainter(surface)
    try:
        canvas.render(painter, QPoint())
    finally:
        painter.end()
    return (time.perf_counter_ns() - started) / 1_000_000.0


def _interaction_phase(
    canvas: DocumentCanvas,
    surface: QImage,
    *,
    action: Callable[[], None],
    cleanup: Callable[[], None] | None = None,
    workload: str,
) -> dict[str, object]:
    """Measure one interaction mutation followed by its synchronous frame.

    The benchmark deliberately does not intercept ``QPainter`` methods.  Such
    monkey-patching changes Python/Qt dispatch and can make the thing being
    measured slower than production.  The phase therefore records the known
    render workload and observed paint-event count instead.
    """

    paint_before = int(getattr(canvas, "paint_event_count", 0))
    started = time.perf_counter_ns()
    action()
    action_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    render_ms = _render_frame(canvas, surface)
    if cleanup is not None:
        cleanup()
    paint_after = int(getattr(canvas, "paint_event_count", 0))
    combined_ms = action_ms + render_ms
    return {
        "applicable": True,
        "action_count": 1,
        "render_count": 1,
        "paint_events_delta": max(0, paint_after - paint_before),
        "workload": workload,
        "action_ms": round(action_ms, 6),
        "render_ms": round(render_ms, 6),
        "combined_ms": round(combined_ms, 6),
    }


def _not_applicable_phase(reason: str) -> dict[str, object]:
    return {
        "applicable": False,
        "action_count": 0,
        "render_count": 0,
        "paint_events_delta": 0,
        "workload": "not_applicable",
        "reason": reason,
    }


def _visible_overlay_phase_set(
    canvas: DocumentCanvas,
) -> tuple[tuple[float, float], ...]:
    keys = canvas._visible_overlay_tile_keys(canvas._paint_context())
    return tuple(
        sorted(
            {
                (
                    float(key.device_phase_x),
                    float(key.device_phase_y),
                )
                for key in keys
            }
        )
    )


def _overlay_payload_summary(canvas: DocumentCanvas) -> dict[str, int]:
    """Describe cached representations without changing hit/miss counters."""

    visible = set(
        canvas._visible_overlay_tile_keys(canvas._paint_context())
    )
    tiles = getattr(canvas_overlay_tile_cache, "_tiles", {})
    summary = {
        "visible_keys": len(visible),
        "available": 0,
        "image_only": 0,
        "picture_only": 0,
        "image_and_picture": 0,
    }
    for key in visible:
        payload = tiles.get(key)
        if payload is None:
            continue
        summary["available"] += 1
        has_image = payload.image is not None
        has_picture = payload.picture is not None
        if has_image and has_picture:
            summary["image_and_picture"] += 1
        elif has_image:
            summary["image_only"] += 1
        elif has_picture:
            summary["picture_only"] += 1
    return summary


def _benchmark_continuous_pan(
    app: QApplication,
    canvas: DocumentCanvas,
    surface: QImage,
    *,
    frames: int,
    overlay_cache_enabled: bool,
) -> dict[str, object]:
    """Measure actual 1-logical-pixel mouse dragging at the canvas DPR.

    Calling the real mouse handlers makes this phase sensitive to the exact
    device-pixel phase used by overlay tile keys.  At 125% scaling an
    unaligned implementation cycles through four phases and falls back to RAW
    vector rendering on three frames out of four; 150% cycles through two.
    """

    if not overlay_cache_enabled:
        return _not_applicable_phase(
            "continuous pan requires the passive overlay cache"
        )
    if isinstance(canvas, DigitalSlideCanvas):
        return _not_applicable_phase(
            "digital-slide navigation has a separate viewport benchmark"
        )
    if frames <= 0:
        return _not_applicable_phase("continuous pan frame count is zero")

    original_pan = Point(canvas._pan.x, canvas._pan.y)
    start = QPointF(canvas.width() * 0.5, canvas.height() * 0.5)
    initial_phase = _visible_overlay_phase_set(canvas)
    cache_before = canvas_overlay_tile_cache.stats()
    paint_before = int(getattr(canvas, "paint_event_count", 0))
    samples: list[float] = []
    frame_misses: list[int] = []
    frame_hits: list[int] = []
    observed_phases: list[tuple[tuple[float, float], ...]] = []

    press = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        start,
        start,
        Qt.MouseButton.MiddleButton,
        Qt.MouseButton.MiddleButton,
        Qt.KeyboardModifier.NoModifier,
    )
    canvas.mousePressEvent(press)
    try:
        for index in range(frames):
            position = QPointF(start.x() + index + 1.0, start.y())
            move = QMouseEvent(
                QEvent.Type.MouseMove,
                position,
                position,
                Qt.MouseButton.NoButton,
                Qt.MouseButton.MiddleButton,
                Qt.KeyboardModifier.NoModifier,
            )
            canvas.mouseMoveEvent(move)
            before_frame = canvas_overlay_tile_cache.stats()
            samples.append(_render_frame(canvas, surface))
            after_frame = canvas_overlay_tile_cache.stats()
            frame_hits.append(max(0, after_frame.hits - before_frame.hits))
            frame_misses.append(
                max(0, after_frame.misses - before_frame.misses)
            )
            observed_phases.append(_visible_overlay_phase_set(canvas))
            # Production receives worker completions through the event loop
            # while the pointer continues to move.
            app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 1)
    finally:
        release_position = QPointF(start.x() + frames, start.y())
        release = QMouseEvent(
            QEvent.Type.MouseButtonRelease,
            release_position,
            release_position,
            Qt.MouseButton.MiddleButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        canvas.mouseReleaseEvent(release)
        canvas._pan = original_pan
        canvas._persist_view_state()

    cache_after = canvas_overlay_tile_cache.stats()
    paint_after = int(getattr(canvas, "paint_event_count", 0))
    phase_change_count = sum(
        phase != initial_phase
        for phase in observed_phases
    )
    timing = _timing_summary(samples)
    return {
        "applicable": True,
        "action_count": frames,
        "render_count": frames,
        "paint_events_delta": max(0, paint_after - paint_before),
        "workload": (
            "continuous 1-logical-pixel middle-button pan using real mouse "
            "handlers and the current device-pixel phase"
        ),
        "logical_step_px": 1.0,
        "device_pixel_ratio": round(
            max(1.0, float(canvas.devicePixelRatioF())),
            4,
        ),
        "initial_phases": [
            {"x": phase[0], "y": phase[1]}
            for phase in initial_phase
        ],
        "observed_phase_sets": [
            [
                {"x": phase[0], "y": phase[1]}
                for phase in phase_set
            ]
            for phase_set in observed_phases
        ],
        "phase_change_count": int(phase_change_count),
        "direct_fallback_frames": sum(
            misses > 0 for misses in frame_misses
        ),
        "cached_frames": sum(
            hits > 0 and misses == 0
            for hits, misses in zip(frame_hits, frame_misses, strict=True)
        ),
        "frame_hits": frame_hits,
        "frame_misses": frame_misses,
        "cache_activity": _cache_activity_delta(
            cache_before,
            cache_after,
        ),
        "timing_ms": timing,
        "combined_ms": round(sum(samples), 6),
        "payloads": _overlay_payload_summary(canvas),
    }


def _canvas_async_work_state(canvas: DocumentCanvas) -> dict[str, int | bool]:
    queue_count, active, scheduled, failed_count = _overlay_wait_state(canvas)
    proxy_timer = getattr(canvas, "_proxy_warm_timer", None)
    return {
        "overlay_queue": int(queue_count),
        "overlay_active": bool(active),
        "overlay_start_scheduled": bool(scheduled),
        "overlay_failed": int(failed_count),
        "proxy_warm_scheduled": bool(
            getattr(canvas, "_proxy_warm_scheduled", False)
        ),
        "proxy_timer_active": bool(
            proxy_timer is not None and proxy_timer.isActive()
        ),
        "tile_pending": int(canvas_overlay_tile_cache.stats().pending),
        "tile_pending_bytes": int(
            canvas_overlay_tile_cache.stats().pending_bytes
        ),
    }


def _settle_visible_canvas(
    app: QApplication,
    canvas: DocumentCanvas,
    *,
    quiet_ms: int = 50,
    timeout_ms: int = 2_000,
) -> dict[str, object]:
    """Drain expected show/update work before measuring unsolicited repaints."""

    started = time.perf_counter()
    deadline = started + (max(1, int(timeout_ms)) / 1000.0)
    quiet_seconds = max(0.001, int(quiet_ms) / 1000.0)
    last_paint_count = int(getattr(canvas, "paint_event_count", 0))
    last_activity_at = time.perf_counter()
    while True:
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 10)
        now = time.perf_counter()
        current_paint_count = int(getattr(canvas, "paint_event_count", 0))
        if current_paint_count != last_paint_count:
            last_paint_count = current_paint_count
            last_activity_at = now
        state = _canvas_async_work_state(canvas)
        no_expected_work = (
            state["overlay_queue"] == 0
            and not state["overlay_active"]
            and not state["overlay_start_scheduled"]
            and not state["proxy_warm_scheduled"]
            and not state["proxy_timer_active"]
            and state["tile_pending"] == 0
            and state["tile_pending_bytes"] == 0
        )
        if no_expected_work and (now - last_activity_at) >= quiet_seconds:
            settled = True
            break
        if now >= deadline:
            settled = False
            break
        time.sleep(0.001)
    return {
        "settled": settled,
        "timed_out": not settled,
        "quiet_ms": int(quiet_ms),
        "timeout_ms": int(timeout_ms),
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "state": _canvas_async_work_state(canvas),
    }


def _idle_observation(
    app: QApplication,
    canvas: DocumentCanvas,
    *,
    duration_ms: int,
) -> dict[str, object]:
    """Observe queued/repeated paint work while the user is idle."""

    paint_before = int(getattr(canvas, "paint_event_count", 0))
    cache_before = canvas_overlay_tile_cache.stats()
    started = time.perf_counter()
    deadline = started + (duration_ms / 1000.0)
    while time.perf_counter() < deadline:
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 5)
        remaining = deadline - time.perf_counter()
        if remaining > 0:
            time.sleep(min(0.001, remaining))
    app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 5)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    paint_after = int(getattr(canvas, "paint_event_count", 0))
    cache_after = canvas_overlay_tile_cache.stats()
    cache_activity = _cache_activity_delta(cache_before, cache_after)
    end_state = _canvas_async_work_state(canvas)
    producers_idle = (
        end_state["overlay_queue"] == 0
        and not end_state["overlay_active"]
        and not end_state["overlay_start_scheduled"]
        and not end_state["proxy_warm_scheduled"]
        and not end_state["proxy_timer_active"]
        and end_state["tile_pending"] == 0
        and end_state["tile_pending_bytes"] == 0
    )
    paint_delta = max(0, paint_after - paint_before)
    return {
        "configured_duration_ms": int(duration_ms),
        "elapsed_ms": round(elapsed_ms, 3),
        "canvas_visible": bool(canvas.isVisible()),
        "paint_events_delta": paint_delta,
        "cache_activity": cache_activity,
        "pending_requests_after": int(cache_after.pending),
        "pending_bytes_after": int(cache_after.pending_bytes),
        "end_state": end_state,
        "quiescent": (
            paint_delta == 0
            and int(cache_activity["hits"]) == 0
            and int(cache_activity["misses"]) == 0
            and int(cache_activity["completed"]) == 0
            and int(cache_activity["defensive_drops"]) == 0
            and producers_idle
        ),
    }


def _benchmark_interactions(
    app: QApplication,
    canvas: DocumentCanvas,
    surface: QImage,
    scenario: ScenarioData,
    *,
    idle_ms: int,
    continuous_pan_frames: int,
    overlay_cache_enabled: bool,
) -> dict[str, object]:
    """Exercise representative user interactions after the hot-frame sample."""

    original_pan = Point(canvas._pan.x, canvas._pan.y)
    original_zoom = float(canvas._zoom)
    original_viewport_origin = (
        canvas.viewport_origin()
        if isinstance(canvas, DigitalSlideCanvas)
        else None
    )

    continuous_pan = _benchmark_continuous_pan(
        app,
        canvas,
        surface,
        frames=continuous_pan_frames,
        overlay_cache_enabled=overlay_cache_enabled,
    )

    def pan_action() -> None:
        if isinstance(canvas, DigitalSlideCanvas):
            canvas.move_viewport_by(12.0, 8.0)
        else:
            canvas._pan = Point(original_pan.x + 12.0, original_pan.y + 8.0)
            canvas._persist_view_state()
            canvas.update()

    def pan_cleanup() -> None:
        if (
            isinstance(canvas, DigitalSlideCanvas)
            and original_viewport_origin is not None
        ):
            current = canvas.viewport_origin()
            canvas.move_viewport_by(
                original_viewport_origin.x - current.x,
                original_viewport_origin.y - current.y,
            )
        else:
            canvas._pan = Point(original_pan.x, original_pan.y)
            canvas._persist_view_state()

    pan = _interaction_phase(
        canvas,
        surface,
        action=pan_action,
        cleanup=pan_cleanup,
        workload="view_transform; passive cache may reuse global tiles",
    )

    def zoom_action() -> None:
        canvas._zoom = max(
            MIN_VIEW_ZOOM,
            min(MAX_VIEW_ZOOM, original_zoom * 1.15),
        )
        canvas._reset_proxy_warming()
        canvas._cancel_overlay_requests()
        canvas._persist_view_state()
        canvas.update()

    def zoom_cleanup() -> None:
        canvas._zoom = original_zoom
        canvas._reset_proxy_warming()
        canvas._cancel_overlay_requests()
        canvas._persist_view_state()

    zoom = _interaction_phase(
        canvas,
        surface,
        action=zoom_action,
        cleanup=zoom_cleanup,
        workload="new exact zoom generation; direct fallback until tiles settle",
    )

    measurement = scenario.document.measurements[0]
    selection = _interaction_phase(
        canvas,
        surface,
        action=lambda: canvas.set_selected_measurement(measurement.id),
        workload="selected object active RAW layer plus passive measurements",
    )

    if measurement.measurement_kind == "area":
        center = area_derived_geometry_service.centroid(measurement)

        def drag_action() -> None:
            canvas._begin_area_drag(
                (measurement.id, "center", None, None),
                center,
            )
            canvas._drag_area_preview_offset = Point(7.0, 5.0)
            canvas.update()

        drag = _interaction_phase(
            canvas,
            surface,
            action=drag_action,
            cleanup=canvas._clear_area_drag_state,
            workload="selected RAW area translated by scalar preview offset",
        )

        def area_point_action() -> None:
            canvas.set_tool_mode("polygon_area")
            position = QPointF(
                max(1.0, min(float(canvas.width() - 2), canvas.width() * 0.5)),
                max(1.0, min(float(canvas.height() - 2), canvas.height() * 0.5)),
            )
            event = QMouseEvent(
                QEvent.Type.MouseButtonPress,
                position,
                position,
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
            canvas.mousePressEvent(event)

        def area_point_cleanup() -> None:
            canvas._cancel_area_drawing()
            canvas.set_tool_mode("select")

        area_point = _interaction_phase(
            canvas,
            surface,
            action=area_point_action,
            cleanup=area_point_cleanup,
            workload="one polygon-area draft point plus exact preview layer",
        )
    else:
        line = measurement.effective_line()

        def drag_action() -> None:
            canvas._dragging_handle = (measurement.id, "end")
            canvas._drag_preview_line = Line(
                line.start,
                Point(line.end.x + 7.0, line.end.y + 5.0),
            )
            canvas.update()

        def drag_cleanup() -> None:
            canvas._dragging_handle = None
            canvas._drag_preview_line = None

        drag = _interaction_phase(
            canvas,
            surface,
            action=drag_action,
            cleanup=drag_cleanup,
            workload="selected line endpoint preview in active layer",
        )
        area_point = _not_applicable_phase("scenario contains no area measurements")

    canvas.set_selected_measurement(None)
    canvas._pan = original_pan
    canvas._zoom = original_zoom
    canvas._persist_view_state()
    # QWidget.update() is ignored for a hidden widget. Show the offscreen
    # canvas, drain the expected show/selection/proxy work, then observe a real
    # visible event loop window. A paint self-loop can no longer hide behind
    # synchronous QWidget.render() calls.
    canvas.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    canvas.show()
    app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 10)
    settle = _settle_visible_canvas(app, canvas)
    idle = _idle_observation(app, canvas, duration_ms=idle_ms)
    idle["settle"] = settle
    idle["valid"] = bool(settle["settled"] and idle["canvas_visible"])
    return {
        "continuous_pan": continuous_pan,
        "pan": pan,
        "zoom": zoom,
        "selection": selection,
        "area_point": area_point,
        "drag": drag,
        "idle": idle,
    }


@contextmanager
def _overlay_cache_environment(enabled: bool):
    """Select one canvas render path without leaking environment changes."""

    enable_name = "FDM_ENABLE_CANVAS_OVERLAY_CACHE"
    disable_name = "FDM_DISABLE_CANVAS_OVERLAY_CACHE"
    previous_enable = os.environ.get(enable_name)
    previous_disable = os.environ.get(disable_name)
    try:
        if enabled:
            os.environ[enable_name] = "1"
            os.environ.pop(disable_name, None)
        else:
            os.environ[disable_name] = "1"
            os.environ.pop(enable_name, None)
        yield
    finally:
        if previous_enable is None:
            os.environ.pop(enable_name, None)
        else:
            os.environ[enable_name] = previous_enable
        if previous_disable is None:
            os.environ.pop(disable_name, None)
        else:
            os.environ[disable_name] = previous_disable


def _cache_activity_delta(
    before: CanvasOverlayCacheStats,
    after: CanvasOverlayCacheStats,
) -> dict[str, int | float]:
    hits = max(0, int(after.hits - before.hits))
    misses = max(0, int(after.misses - before.misses))
    requests = hits + misses
    defensive_drops = max(0, int(after.dropped - before.dropped))
    return {
        "hits": hits,
        "misses": misses,
        "hit_rate": round(hits / requests, 6) if requests else 0.0,
        "completed": max(0, int(after.completed - before.completed)),
        # Production exposes one aggregate defensive count. The benchmark
        # reports generation-late results separately through its completion
        # observer and keeps this field accurately named.
        "defensive_drops": defensive_drops,
        # Preserve the schema-v1 activity key for existing result consumers.
        "late_or_rejected": defensive_drops,
    }


def _overlay_wait_state(canvas: DocumentCanvas) -> tuple[int, bool, bool, int]:
    queue = getattr(canvas, "_overlay_tile_queue", ())
    active = getattr(canvas, "_overlay_tile_active", None) is not None
    scheduled = bool(getattr(canvas, "_overlay_tile_build_scheduled", False))
    failed = getattr(canvas, "_overlay_tile_failed", ())
    return len(queue), active, scheduled, len(failed)


def _wait_for_overlay_tiles(
    app: QApplication,
    canvas: DocumentCanvas,
    *,
    baseline: CanvasOverlayCacheStats,
    requested_tile_count: int,
    timeout_ms: int,
) -> dict[str, object]:
    """Pump Qt events until the visible tile queue settles or times out."""

    started = time.perf_counter()
    deadline = started + (timeout_ms / 1000.0)
    ready = False
    failed_tiles = 0
    while True:
        # Worker completion is delivered through queued Qt signals. Explicitly
        # processing events is therefore part of the benchmark contract.
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 10)
        queue_count, active, scheduled, failed_tiles = _overlay_wait_state(canvas)
        stats = canvas_overlay_tile_cache.stats()
        completed = max(0, stats.completed - baseline.completed)
        settled = (
            queue_count == 0
            and not active
            and not scheduled
            and stats.pending == 0
        )
        ready = (
            settled
            and failed_tiles == 0
            and completed >= requested_tile_count
        )
        if ready or time.perf_counter() >= deadline:
            break
        # Do not spin at 100% CPU while the worker rasterizes a tile.
        time.sleep(0.001)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    queue_count, active, scheduled, failed_tiles = _overlay_wait_state(canvas)
    final_stats = canvas_overlay_tile_cache.stats()
    return {
        "requested_tiles": int(requested_tile_count),
        "ready": ready,
        "timed_out": not ready,
        "timeout_ms": int(timeout_ms),
        "elapsed_ms": round(elapsed_ms, 3),
        "remaining_queue": int(queue_count),
        "active_request": bool(active),
        "start_scheduled": bool(scheduled),
        "pending_requests": int(final_stats.pending),
        "pending_bytes": int(final_stats.pending_bytes),
        "failed_tiles": int(failed_tiles),
    }


def _ensure_application() -> QApplication:
    existing = QApplication.instance()
    if existing is not None:
        return existing
    app = QApplication(["fdm-canvas-benchmark", "-platform", os.environ["QT_QPA_PLATFORM"]])
    app.setQuitOnLastWindowClosed(False)
    return app


def _set_benchmark_document(
    canvas: DocumentCanvas,
    scenario: ScenarioData,
    *,
    canvas_kind: str,
) -> _BenchmarkDigitalSlideStore | None:
    if canvas_kind != "digital_slide":
        canvas.set_document(scenario.document, scenario.image)
        if scenario.definition.family == "magic_area":
            canvas.set_tool_mode(MagicSegmentToolMode.STANDARD)
        return None

    assert isinstance(canvas, _BenchmarkDigitalSlideCanvas)
    origin_x = 128
    origin_y = 96
    scenario.document.metadata["digital_slide"] = {
        "viewport_origin": [origin_x, origin_y],
        "focus_index": 0,
    }
    manifest = DigitalSlideManifest(
        version=1,
        width=scenario.image.width() * 4,
        height=scenario.image.height() * 4,
        viewport_width=scenario.image.width(),
        viewport_height=scenario.image.height(),
        focus_levels=[0],
        tile_count=1,
        status="ready",
        metadata={"benchmark_fixture": True},
    )
    store = _BenchmarkDigitalSlideStore(
        manifest=manifest,
        fill_color=QColor("#25313C"),
    )
    canvas.set_slide_document(scenario.document, store)
    if scenario.definition.family == "magic_area":
        canvas.set_tool_mode(MagicSegmentToolMode.STANDARD)
    # ``set_slide_document()`` intentionally schedules the first fit on the
    # event loop so a real tab can obtain its final size.  The benchmark has
    # already assigned a deterministic size; settle that pending transition
    # before the cold frame so its queued callback cannot invalidate the
    # overlay-cache generation while the readiness probe is waiting.
    canvas._apply_initial_fit()  # noqa: SLF001
    return store


def _wait_for_digital_slide_generation(
    app: QApplication,
    canvas: _BenchmarkDigitalSlideCanvas,
    generation: int,
    *,
    timeout_seconds: float = 5.0,
) -> float:
    started = time.perf_counter()
    deadline = started + timeout_seconds
    while time.perf_counter() < deadline:
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 10)
        frame = canvas._render_frame  # noqa: SLF001 - benchmark probe
        if (
            frame is not None
            and frame.generation == int(generation)
            and frame.quality == "final"
            and (
                canvas.large_area_browse_active()
                or frame.pixel_exact
            )
        ):
            break
        time.sleep(0.001)
    return (time.perf_counter() - started) * 1000.0


def _benchmark_digital_slide_camera(
    app: QApplication,
    canvas: _BenchmarkDigitalSlideCanvas,
) -> dict[str, object]:
    """Exercise real 100/50/25/whole-slide LODs and all eight directions."""

    canvas.fit_native_viewport()
    native_zoom = float(canvas.view_zoom())
    zoom_phases: dict[str, object] = {}
    for name, ratio in (("100", 1.0), ("50", 0.5), ("25", 0.25)):
        started = time.perf_counter()
        canvas.set_view_zoom(native_zoom * ratio)
        input_ms = (time.perf_counter() - started) * 1000.0
        generation = int(canvas._view_generation)  # noqa: SLF001
        final_ms = _wait_for_digital_slide_generation(app, canvas, generation)
        frame = canvas._render_frame  # noqa: SLF001
        visible = canvas.visible_slide_rect()
        zoom_phases[name] = {
            "input_ms": round(input_ms, 3),
            "final_frame_ms": round(final_ms, 3),
            "visible_width": float(visible.width()),
            "visible_height": float(visible.height()),
            "lod": int(frame.lod) if frame is not None else None,
        }
    started = time.perf_counter()
    canvas.fit_to_view()
    input_ms = (time.perf_counter() - started) * 1000.0
    generation = int(canvas._view_generation)  # noqa: SLF001
    final_ms = _wait_for_digital_slide_generation(app, canvas, generation)
    frame = canvas._render_frame  # noqa: SLF001
    visible = canvas.visible_slide_rect()
    zoom_phases["whole"] = {
        "input_ms": round(input_ms, 3),
        "final_frame_ms": round(final_ms, 3),
        "visible_width": float(visible.width()),
        "visible_height": float(visible.height()),
        "lod": int(frame.lod) if frame is not None else None,
    }

    directions = {
        "left": (-1.0, 0.0),
        "right": (1.0, 0.0),
        "up": (0.0, -1.0),
        "down": (0.0, 1.0),
        "up_left": (-1.0, -1.0),
        "up_right": (1.0, -1.0),
        "down_left": (-1.0, 1.0),
        "down_right": (1.0, 1.0),
    }
    navigation: dict[str, object] = {}
    fast_navigation: dict[str, object] = {}
    manifest = canvas._slide_manifest  # noqa: SLF001
    assert manifest is not None
    for result, fraction in ((navigation, 0.25), (fast_navigation, 1.0)):
        for name, (unit_x, unit_y) in directions.items():
            canvas.fit_native_viewport()
            canvas.center_on_image_point(
                Point(float(manifest.width) / 2.0, float(manifest.height) / 2.0)
            )
            source = canvas._source_view_rect()  # noqa: SLF001
            dx = source.width() * fraction * unit_x
            dy = source.height() * fraction * unit_y
            if dx and dy:
                dx /= math.sqrt(2.0)
                dy /= math.sqrt(2.0)
            started = time.perf_counter()
            canvas.move_viewport_by(dx, dy)
            input_ms = (time.perf_counter() - started) * 1000.0
            generation = int(canvas._view_generation)  # noqa: SLF001
            final_ms = _wait_for_digital_slide_generation(
                app,
                canvas,
                generation,
            )
            result[name] = {
                "input_ms": round(input_ms, 3),
                "final_frame_ms": round(final_ms, 3),
            }

    canvas.fit_native_viewport()
    generation = int(canvas._view_generation)  # noqa: SLF001
    _wait_for_digital_slide_generation(app, canvas, generation)
    return {
        "zoom_levels": zoom_phases,
        "directions": navigation,
        "fast_directions": fast_navigation,
    }


def run_benchmark(
    scenario_name: str,
    *,
    object_count: int | None = None,
    coordinate_count: int | None = None,
    frames: int = 8,
    warmup_frames: int = 2,
    canvas_size: tuple[int, int] = DEFAULT_CANVAS_SIZE,
    seed: int = DEFAULT_SEED,
    overlay_cache: bool = False,
    overlay_cache_timeout_ms: int = 5_000,
    canvas_kind: str = "document",
    idle_ms: int = 500,
    device_pixel_ratio: float = 1.0,
    continuous_pan_frames: int = 12,
) -> dict[str, object]:
    """Run a scenario and return its versioned, JSON-serializable result."""

    if frames <= 0:
        raise ValueError("frames must be greater than zero")
    if warmup_frames < 0:
        raise ValueError("warmup_frames cannot be negative")
    if overlay_cache_timeout_ms <= 0:
        raise ValueError("overlay_cache_timeout_ms must be greater than zero")
    if canvas_kind not in {"document", "digital_slide"}:
        raise ValueError("canvas_kind must be 'document' or 'digital_slide'")
    if idle_ms < 0:
        raise ValueError("idle_ms cannot be negative")
    if (
        not math.isfinite(float(device_pixel_ratio))
        or float(device_pixel_ratio) < 1.0
        or float(device_pixel_ratio) > 4.0
    ):
        raise ValueError("device_pixel_ratio must be between 1.0 and 4.0")
    if continuous_pan_frames < 0:
        raise ValueError("continuous_pan_frames cannot be negative")

    app = _ensure_application()
    gc.collect()
    rss_before_build, rss_provider = _current_rss_bytes()
    scenario = build_scenario(
        scenario_name,
        object_count=object_count,
        coordinate_count=coordinate_count,
        canvas_size=canvas_size,
        seed=seed,
    )
    cache_reset = _reset_runtime_caches(app)
    cache_before = canvas_overlay_tile_cache.stats()
    with _overlay_cache_environment(overlay_cache):
        canvas = (
            _BenchmarkDigitalSlideCanvas(
                device_pixel_ratio=device_pixel_ratio,
            )
            if canvas_kind == "digital_slide"
            else _BenchmarkCanvas(
                device_pixel_ratio=device_pixel_ratio,
            )
        )
        canvas.resize(*canvas_size)
        if canvas_kind == "digital_slide":
            scenario.settings.digital_slide_render_cache_gib = 0
        canvas.set_settings(scenario.settings)
        digital_slide_store = _set_benchmark_document(
            canvas,
            scenario,
            canvas_kind=canvas_kind,
        )
        runtime_cache_before = _runtime_cache_snapshot()
        surface_dpr = max(1.0, float(device_pixel_ratio))
        surface = QImage(
            max(1, int(math.ceil(canvas_size[0] * surface_dpr))),
            max(1, int(math.ceil(canvas_size[1] * surface_dpr))),
            QImage.Format.Format_ARGB32_Premultiplied,
        )
        surface.setDevicePixelRatio(surface_dpr)
        rss_after_setup, setup_rss_provider = _current_rss_bytes()
        if rss_provider is None:
            rss_provider = setup_rss_provider
        if isinstance(canvas, _BenchmarkDigitalSlideCanvas):
            canvas.show()
            deadline = time.perf_counter() + 5.0
            while (
                canvas._render_frame is None  # noqa: SLF001 - benchmark probe
                and time.perf_counter() < deadline
            ):
                app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 10)
                time.sleep(0.001)
            digital_slide_camera_benchmark = _benchmark_digital_slide_camera(
                app,
                canvas,
            )
        else:
            digital_slide_camera_benchmark = None

        try:
            with _trace_overlay_drop_reasons(canvas) as drop_reasons:
                cold_samples = [_render_frame(canvas, surface)]
                cache_after_cold = canvas_overlay_tile_cache.stats()
                runtime_cache_after_cold = _runtime_cache_snapshot()
                queued, active, _scheduled, _failed = _overlay_wait_state(canvas)
                requested_tiles = queued + int(active)
                if overlay_cache:
                    overlay_wait = _wait_for_overlay_tiles(
                        app,
                        canvas,
                        baseline=cache_before,
                        requested_tile_count=requested_tiles,
                        timeout_ms=overlay_cache_timeout_ms,
                    )
                else:
                    overlay_wait = {
                        "requested_tiles": 0,
                        "ready": False,
                        "timed_out": False,
                        "timeout_ms": int(overlay_cache_timeout_ms),
                        "elapsed_ms": 0.0,
                        "remaining_queue": 0,
                        "active_request": False,
                        "start_scheduled": False,
                        "pending_requests": 0,
                        "pending_bytes": 0,
                        "failed_tiles": 0,
                    }
                cache_after_wait = canvas_overlay_tile_cache.stats()
                for _ in range(warmup_frames):
                    _render_frame(canvas, surface)
                cache_before_hot = canvas_overlay_tile_cache.stats()
                hot_samples = [
                    _render_frame(canvas, surface) for _ in range(frames)
                ]
                cache_after_hot = canvas_overlay_tile_cache.stats()
                runtime_cache_after_hot = _runtime_cache_snapshot()
                painter_call_trace = _render_call_trace(canvas, surface)
                interactions = _benchmark_interactions(
                    app,
                    canvas,
                    surface,
                    scenario,
                    idle_ms=idle_ms,
                    continuous_pan_frames=continuous_pan_frames,
                    overlay_cache_enabled=overlay_cache,
                )
                cache_after_interactions = canvas_overlay_tile_cache.stats()
                runtime_cache_after_interactions = _runtime_cache_snapshot()
                rss_after_frames, frames_rss_provider = _current_rss_bytes()
                if rss_provider is None:
                    rss_provider = frames_rss_provider
                peak_rss, peak_rss_provider = _peak_rss_bytes()
                all_samples = [*cold_samples, *hot_samples]
                cache_ready = bool(overlay_wait["ready"])
                requested_path = (
                    "overlay_cache" if overlay_cache else "direct"
                )
                effective_hot_path = (
                    "overlay_cache"
                    if overlay_cache and cache_ready
                    else "overlay_cache_timeout_fallback"
                    if overlay_cache
                    else "direct"
                )
                total_cache_activity = _cache_activity_delta(
                    cache_before,
                    cache_after_interactions,
                )
                aggregate_defensive_drops = int(
                    total_cache_activity["defensive_drops"]
                )
                generation_late_drops = min(
                    aggregate_defensive_drops,
                    int(drop_reasons.generation_late),
                )
                other_defensive_drops = max(
                    0,
                    aggregate_defensive_drops - generation_late_drops,
                )
                interaction_render_count = sum(
                    int(phase.get("render_count", 0))
                    for name, phase in interactions.items()
                    if name != "idle" and isinstance(phase, dict)
                )
                measured_render_count = (
                    len(cold_samples)
                    + len(hot_samples)
                    + interaction_render_count
                )
                classified_cached_frames = (
                    len(hot_samples) if cache_ready else 0
                )
                classified_direct_frames = len(cold_samples) + (
                    0 if cache_ready else len(hot_samples)
                )
                digital_slide_payload: dict[str, object] | None = None
                if isinstance(canvas, _BenchmarkDigitalSlideCanvas):
                    viewport_origin = canvas.viewport_origin()
                    renderer_stats = canvas.renderer_stats()
                    digital_slide_payload = {
                        "set_slide_document_used": True,
                        "store_kind": "temporary_sqlite_png_jpeg",
                        "viewport_origin": {
                            "x": float(viewport_origin.x),
                            "y": float(viewport_origin.y),
                        },
                        "viewport_buffer_requests": int(
                            canvas.viewport_buffer_request_count
                        ),
                        "renderer": (
                            {
                                "submitted": renderer_stats.submitted,
                                "completed": renderer_stats.completed,
                                "cancelled": renderer_stats.cancelled,
                                "stale_dropped": renderer_stats.stale_dropped,
                                "decoded_tiles": renderer_stats.decoded_tiles,
                                "memory_hits": renderer_stats.memory_hits,
                                "disk_hits": renderer_stats.disk_hits,
                                "memory_bytes": renderer_stats.memory_bytes,
                                "pending_requests": renderer_stats.pending_requests,
                            }
                            if renderer_stats is not None
                            else None
                        ),
                        "camera_benchmark": digital_slide_camera_benchmark,
                    }
                result: dict[str, object] = {
                    "schema_version": SCHEMA_VERSION,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "scenario": {
                        "name": scenario.definition.name,
                        "family": scenario.definition.family,
                        "description": scenario.definition.description,
                        "object_count": scenario.object_count,
                        "coordinate_count": scenario.coordinate_count,
                        "labels_enabled": scenario.definition.labels_enabled,
                        "composition": scenario.definition.composition,
                        "seed": scenario.seed,
                        "image_size": {
                            "width": scenario.image.width(),
                            "height": scenario.image.height(),
                        },
                        "canvas_size": {
                            "width": canvas_size[0],
                            "height": canvas_size[1],
                        },
                        "canvas_kind": canvas_kind,
                        "device_pixel_ratio": round(
                            float(device_pixel_ratio),
                            4,
                        ),
                        "digital_slide": digital_slide_payload,
                    },
                    "render_path": {
                        "requested": requested_path,
                        "effective_hot": effective_hot_path,
                        "cache_ready": cache_ready,
                    },
                    "timing_ms": {
                        "cold": _timing_summary(cold_samples),
                        "hot": _timing_summary(hot_samples),
                        "all_measured": _timing_summary(all_samples),
                        "warmup_frame_count": warmup_frames,
                    },
                    "overlay_cache": {
                        "enabled": bool(overlay_cache),
                        "wait": overlay_wait,
                        "tiles": {
                            "entries": int(cache_after_hot.entries),
                            "bytes": int(cache_after_hot.bytes),
                            "pending": int(cache_after_hot.pending),
                            "pending_bytes": int(
                                cache_after_hot.pending_bytes
                            ),
                        },
                        "cold_activity": _cache_activity_delta(
                            cache_before,
                            cache_after_cold,
                        ),
                        "warm_activity": _cache_activity_delta(
                            cache_after_cold,
                            cache_after_wait,
                        ),
                        "hot_activity": _cache_activity_delta(
                            cache_before_hot,
                            cache_after_hot,
                        ),
                        "total_activity": total_cache_activity,
                        "defensive_drop_count": aggregate_defensive_drops,
                        "generation_late_drop_count": generation_late_drops,
                        "other_defensive_drop_count": other_defensive_drops,
                        # Preserve the schema-v1 aggregate compatibility alias.
                        "late_drop_count": aggregate_defensive_drops,
                    },
                    "runtime_caches": {
                        "cold_reset": cache_reset,
                        "before_render": runtime_cache_before,
                        "after_cold": runtime_cache_after_cold,
                        "after_hot": runtime_cache_after_hot,
                        "after_interactions": (
                            runtime_cache_after_interactions
                        ),
                        "activity": {
                            "label_sprites": _cache_counter_delta(
                                runtime_cache_before,
                                runtime_cache_after_interactions,
                                cache_name="label_sprites",
                                fields=("hits", "misses", "evictions"),
                            ),
                            "area_handles": _cache_counter_delta(
                                runtime_cache_before,
                                runtime_cache_after_interactions,
                                cache_name="area_handles",
                                fields=("hits", "misses", "evictions"),
                            ),
                        },
                    },
                    "interactions": interactions,
                    "render_workload": {
                        "instrumentation": (
                            "timing uses unmodified QPainter; one separate "
                            "UI-thread trace frame counts explicit "
                            "drawPath/drawImage calls"
                        ),
                        "draw_calls": painter_call_trace,
                        "measured_render_calls": measured_render_count,
                        "classified_direct_frames": (
                            classified_direct_frames
                        ),
                        "classified_cached_frames": (
                            classified_cached_frames
                        ),
                        "interaction_frames": interaction_render_count,
                        "interaction_classification": (
                            "mixed_or_direct because selection and exact zoom "
                            "can invalidate or bypass passive tiles"
                        ),
                    },
                    "paint_events": {
                        "total": canvas.paint_event_count,
                        # Preserve the v1 meaning: cold + hot timing samples.
                        "measured": len(all_samples),
                        "warmup": warmup_frames,
                        "trace_render_count": 1,
                        "interaction_render_count": interaction_render_count,
                    },
                    "rss": {
                        "available": any(
                            value is not None
                            for value in (
                                rss_before_build,
                                rss_after_setup,
                                rss_after_frames,
                                peak_rss,
                            )
                        ),
                        "before_build_bytes": rss_before_build,
                        "after_setup_bytes": rss_after_setup,
                        "after_frames_bytes": rss_after_frames,
                        "delta_bytes": (
                            rss_after_frames - rss_before_build
                            if rss_after_frames is not None
                            and rss_before_build is not None
                            else None
                        ),
                        "peak_bytes": peak_rss,
                        "current_provider": rss_provider,
                        "peak_provider": peak_rss_provider,
                    },
                    "environment": _environment_payload(app, canvas=canvas),
                }
                return result
        finally:
            if isinstance(canvas, DigitalSlideCanvas):
                canvas.shutdown()
            if digital_slide_store is not None:
                digital_slide_store.close()
            canvas.clear_document()
            canvas.close()
            canvas.deleteLater()
            canvas_overlay_tile_cache.invalidate_document(id(scenario.document))
            app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 10)
            _reset_runtime_caches(app, drain_timeout_ms=250)


def _resolve_output_path(raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = BENCHMARK_OUTPUT_ROOT / candidate
    # Suffix normalization must happen before containment validation. Checking
    # ``.../.tmp`` first and then converting it to ``.../.tmp.json`` would
    # otherwise move the final file outside the ignored directory.
    if candidate.suffix.lower() != ".json":
        candidate = candidate.with_suffix(".json")
    resolved = candidate.resolve()
    ignored_root = (PROJECT_ROOT / ".tmp").resolve()
    if resolved != ignored_root and ignored_root not in resolved.parents:
        raise ValueError(
            f"benchmark output must stay inside the ignored directory {ignored_root}"
        )
    return resolved


def _human_summary(result: dict[str, object]) -> str:
    scenario = result["scenario"]
    render_path = result["render_path"]
    timing = result["timing_ms"]
    overlay_cache = result["overlay_cache"]
    rss = result["rss"]
    assert isinstance(scenario, dict)
    assert isinstance(render_path, dict)
    assert isinstance(timing, dict)
    assert isinstance(overlay_cache, dict)
    assert isinstance(rss, dict)
    cold = timing["cold"]
    hot = timing["hot"]
    assert isinstance(cold, dict)
    assert isinstance(hot, dict)
    cache_tiles = overlay_cache["tiles"]
    cache_hot = overlay_cache["hot_activity"]
    cache_wait = overlay_cache["wait"]
    interactions = result["interactions"]
    assert isinstance(cache_tiles, dict)
    assert isinstance(cache_hot, dict)
    assert isinstance(cache_wait, dict)
    assert isinstance(interactions, dict)
    continuous_pan = interactions.get("continuous_pan", {})
    assert isinstance(continuous_pan, dict)
    rss_after = rss.get("after_frames_bytes")
    rss_text = (
        f"{int(rss_after) / (1024 * 1024):.1f} MiB"
        if isinstance(rss_after, int)
        else "unavailable"
    )
    return "\n".join(
        (
            f"Scenario: {scenario['name']}",
            (
                f"Objects: {scenario['object_count']}  "
                f"Coordinates: {scenario['coordinate_count']}  "
                f"Labels: {scenario['labels_enabled']}"
            ),
            (
                f"Render path: requested={render_path['requested']}  "
                f"effective-hot={render_path['effective_hot']}"
            ),
            (
                f"Cold frame: P50 {cold['p50']:.3f} ms  "
                f"P95 {cold['p95']:.3f} ms  max {cold['max']:.3f} ms"
            ),
            (
                f"Hot frames: P50 {hot['p50']:.3f} ms  "
                f"P95 {hot['p95']:.3f} ms  max {hot['max']:.3f} ms"
            ),
            (
                f"Overlay cache: ready={cache_wait['ready']}  "
                f"tiles={cache_tiles['entries']}  bytes={cache_tiles['bytes']}  "
                f"hot-hit-rate={float(cache_hot['hit_rate']):.3f}  "
                f"late/drop={overlay_cache['late_drop_count']}"
            ),
            (
                "Continuous pan: "
                f"applicable={continuous_pan.get('applicable', False)}  "
                f"DPR={continuous_pan.get('device_pixel_ratio', 'n/a')}  "
                f"phase-changes={continuous_pan.get('phase_change_count', 0)}  "
                f"direct-fallback-frames="
                f"{continuous_pan.get('direct_fallback_frames', 0)}"
            ),
            f"RSS after frames: {rss_text}",
        )
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deterministic offscreen measurement-canvas benchmarks."
    )
    parser.add_argument("--scenario", choices=sorted(SCENARIOS))
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List available scenarios and exit.",
    )
    parser.add_argument(
        "--objects",
        type=int,
        help="Override the scenario's default object count.",
    )
    parser.add_argument(
        "--coordinates",
        type=int,
        help="Override total RAW coordinates for an area scenario.",
    )
    parser.add_argument("--frames", type=int, default=8, help="Measured hot frames.")
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Unmeasured frames between the cold frame and hot samples.",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_CANVAS_SIZE[0])
    parser.add_argument("--height", type=int, default=DEFAULT_CANVAS_SIZE[1])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--overlay-cache",
        action="store_true",
        help=(
            "Warm and measure the exact passive overlay-tile cache. The "
            "default measures the direct vector/sprite path."
        ),
    )
    parser.add_argument(
        "--overlay-cache-timeout-ms",
        type=int,
        default=5_000,
        help="Maximum Qt-event-pumped wait for visible overlay tiles.",
    )
    parser.add_argument(
        "--canvas-kind",
        choices=("document", "digital_slide"),
        default="document",
        help="Benchmark the ordinary image canvas or digital-slide canvas.",
    )
    parser.add_argument(
        "--idle-ms",
        type=int,
        default=500,
        help="Idle observation window used to detect unsolicited repaints.",
    )
    parser.add_argument(
        "--device-pixel-ratio",
        type=float,
        default=1.0,
        help=(
            "Deterministic canvas DPR override; use 1.25 or 1.5 to reproduce "
            "common Windows scaling."
        ),
    )
    parser.add_argument(
        "--pan-frames",
        type=int,
        default=12,
        help=(
            "Continuous one-logical-pixel pan frames measured after overlay "
            "cache warm-up."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete machine-readable result.",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help=(
            "Also write JSON under the ignored .tmp directory. Relative paths "
            "are placed in .tmp/canvas-benchmark/."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.list_scenarios:
        for name in sorted(SCENARIOS):
            definition = SCENARIOS[name]
            print(f"{name}: {definition.description}")
        return 0
    if not arguments.scenario:
        parser.error("--scenario is required unless --list-scenarios is used")
    try:
        result = run_benchmark(
            arguments.scenario,
            object_count=arguments.objects,
            coordinate_count=arguments.coordinates,
            frames=arguments.frames,
            warmup_frames=arguments.warmup,
            canvas_size=(arguments.width, arguments.height),
            seed=arguments.seed,
            overlay_cache=arguments.overlay_cache,
            overlay_cache_timeout_ms=arguments.overlay_cache_timeout_ms,
            canvas_kind=arguments.canvas_kind,
            idle_ms=arguments.idle_ms,
            device_pixel_ratio=arguments.device_pixel_ratio,
            continuous_pan_frames=arguments.pan_frames,
        )
        encoded = json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        if arguments.output:
            output_path = _resolve_output_path(arguments.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(f"{encoded}\n", encoding="utf-8")
        print(encoded if arguments.json else _human_summary(result))
        return 0
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
