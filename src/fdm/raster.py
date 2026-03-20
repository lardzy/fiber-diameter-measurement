from __future__ import annotations

from dataclasses import dataclass
import math

from fdm.geometry import Line, Point, direction, midpoint, normal


@dataclass(slots=True)
class RasterImage:
    width: int
    height: int
    pixels: list[int]

    @classmethod
    def blank(cls, width: int, height: int, fill: int = 255) -> "RasterImage":
        return cls(width=width, height=height, pixels=[fill] * (width * height))

    @classmethod
    def from_rows(cls, rows: list[list[int]]) -> "RasterImage":
        if not rows:
            return cls(width=0, height=0, pixels=[])
        height = len(rows)
        width = len(rows[0])
        pixels: list[int] = []
        for row in rows:
            if len(row) != width:
                raise ValueError("All rows must have the same width.")
            pixels.extend(int(max(0, min(255, value))) for value in row)
        return cls(width=width, height=height, pixels=pixels)

    def index(self, x: int, y: int) -> int:
        return y * self.width + x

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def get(self, x: int, y: int, default: int = 255) -> int:
        if not self.in_bounds(x, y):
            return default
        return self.pixels[self.index(x, y)]

    def set(self, x: int, y: int, value: int) -> None:
        if self.in_bounds(x, y):
            self.pixels[self.index(x, y)] = int(max(0, min(255, value)))

    def sample(self, x: float, y: float, default: int = 255) -> int:
        return self.get(int(round(x)), int(round(y)), default=default)

    def to_rows(self) -> list[list[int]]:
        return [
            self.pixels[row_start:row_start + self.width]
            for row_start in range(0, len(self.pixels), self.width)
        ]

    def mean(self) -> float:
        if not self.pixels:
            return 0.0
        return sum(self.pixels) / len(self.pixels)

    def stddev(self) -> float:
        if not self.pixels:
            return 0.0
        mean_value = self.mean()
        variance = sum((value - mean_value) ** 2 for value in self.pixels) / len(self.pixels)
        return math.sqrt(variance)


@dataclass(slots=True)
class RotatedROI:
    image: RasterImage
    center: Point
    axis_x: tuple[float, float]
    axis_y: tuple[float, float]
    source_line: Line
    width: int
    height: int

    @property
    def midpoint(self) -> Point:
        return Point(self.width / 2.0, self.height / 2.0)

    def map_roi_to_image(self, point: Point) -> Point:
        dx = point.x - self.width / 2.0
        dy = point.y - self.height / 2.0
        return Point(
            x=self.center.x + self.axis_x[0] * dx + self.axis_y[0] * dy,
            y=self.center.y + self.axis_x[1] * dx + self.axis_y[1] * dy,
        )

    def map_image_to_roi(self, point: Point) -> Point:
        dx = point.x - self.center.x
        dy = point.y - self.center.y
        return Point(
            x=dx * self.axis_x[0] + dy * self.axis_x[1] + self.width / 2.0,
            y=dx * self.axis_y[0] + dy * self.axis_y[1] + self.height / 2.0,
        )


def extract_rotated_roi(
    image: RasterImage,
    line: Line,
    *,
    padding: int = 48,
    half_height: int = 64,
) -> RotatedROI:
    axis_x = direction(line)
    axis_y = normal(axis_x)
    line_midpoint = midpoint(line)
    line_width = max(8, int(math.ceil(math.hypot(line.end.x - line.start.x, line.end.y - line.start.y))))
    roi_width = line_width + padding * 2
    roi_height = half_height * 2
    background = int(round(image.mean())) if image.pixels else 255
    roi_image = RasterImage.blank(roi_width, roi_height, fill=background)
    roi = RotatedROI(
        image=roi_image,
        center=line_midpoint,
        axis_x=axis_x,
        axis_y=axis_y,
        source_line=line,
        width=roi_width,
        height=roi_height,
    )
    for y in range(roi_height):
        for x in range(roi_width):
            source = roi.map_roi_to_image(Point(float(x), float(y)))
            roi_image.set(x, y, image.sample(source.x, source.y, default=background))
    return roi
