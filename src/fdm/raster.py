from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import math

from fdm.geometry import Line, Point, direction, midpoint, normal


class RasterPixelType(str, Enum):
    """Canonical pixel layouts supported by the image-processing boundary.

    ``RasterImage`` below remains the mutable 8-bit grayscale helper used by
    the existing edge-snap pipeline.  This enum and :class:`RasterPlane` add a
    typed, immutable boundary without changing that legacy call path.

    Multi-byte scalar samples use little-endian byte order.  RGB(A) samples
    are tightly packed in channel order.
    """

    GRAY8 = "gray8"
    GRAY16 = "gray16"
    GRAY32_FLOAT = "gray32_float"
    RGB8 = "rgb8"
    RGBA8 = "rgba8"

    @property
    def channel_count(self) -> int:
        return {
            RasterPixelType.GRAY8: 1,
            RasterPixelType.GRAY16: 1,
            RasterPixelType.GRAY32_FLOAT: 1,
            RasterPixelType.RGB8: 3,
            RasterPixelType.RGBA8: 4,
        }[self]

    @property
    def bytes_per_channel(self) -> int:
        if self is RasterPixelType.GRAY16:
            return 2
        if self is RasterPixelType.GRAY32_FLOAT:
            return 4
        return 1

    @property
    def bytes_per_pixel(self) -> int:
        return self.channel_count * self.bytes_per_channel

    @property
    def sample_maximum(self) -> int | None:
        if self is RasterPixelType.GRAY16:
            return 65_535
        if self is RasterPixelType.GRAY32_FLOAT:
            return None
        return 255

    @property
    def is_grayscale(self) -> bool:
        return self in {
            RasterPixelType.GRAY8,
            RasterPixelType.GRAY16,
            RasterPixelType.GRAY32_FLOAT,
        }

    @property
    def has_alpha(self) -> bool:
        return self is RasterPixelType.RGBA8

    @classmethod
    def parse(cls, value: object) -> "RasterPixelType":
        if isinstance(value, cls):
            return value
        token = str(value or "").strip().lower()
        try:
            return cls(token)
        except ValueError as exc:
            supported = "、".join(item.value for item in cls)
            raise ValueError(f"不支持的栅格像素类型: {value!r}；支持 {supported}") from exc


@dataclass(frozen=True, slots=True)
class RasterPlane:
    """An immutable, tightly packed raster snapshot.

    Pixel bytes deliberately stay out of project JSON.  Project persistence
    stores a lossless image asset plus the small pixel-type/derivation
    descriptors on ``ImageDocument``.  Keeping this value object byte-backed
    makes it safe to hand to worker threads and prevents a mutable NumPy view
    from changing beneath a generation-checked request.
    """

    width: int
    height: int
    pixel_type: RasterPixelType
    data: bytes

    def __post_init__(self) -> None:
        width = _dimension_value(self.width, field_name="width")
        height = _dimension_value(self.height, field_name="height")
        if (width == 0) != (height == 0):
            raise ValueError("空栅格的 width 和 height 必须同时为 0")
        pixel_type = RasterPixelType.parse(self.pixel_type)
        try:
            data = bytes(self.data)
        except (TypeError, ValueError) as exc:
            raise TypeError("RasterPlane.data 必须是 bytes-like 对象") from exc
        expected = width * height * pixel_type.bytes_per_pixel
        if len(data) != expected:
            raise ValueError(
                f"栅格字节数不匹配: 期望 {expected}，实际 {len(data)}"
            )
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "pixel_type", pixel_type)
        object.__setattr__(self, "data", data)

    @property
    def byte_count(self) -> int:
        return len(self.data)

    @property
    def row_bytes(self) -> int:
        return self.width * self.pixel_type.bytes_per_pixel

    @property
    def is_empty(self) -> bool:
        return self.width == 0

    def sha256(self) -> str:
        """Return a content identity including layout and dimensions."""

        digest = hashlib.sha256()
        digest.update(b"fdm-raster-plane-v1\0")
        digest.update(self.pixel_type.value.encode("ascii"))
        digest.update(b"\0")
        digest.update(self.width.to_bytes(8, "little", signed=False))
        digest.update(self.height.to_bytes(8, "little", signed=False))
        digest.update(self.data)
        return digest.hexdigest()

    @classmethod
    def from_raster_image(cls, image: "RasterImage") -> "RasterPlane":
        pixels = tuple(int(value) for value in image.pixels)
        if any(value < 0 or value > 255 for value in pixels):
            raise ValueError("RasterImage 包含超出 8 位范围的像素")
        return cls(
            width=image.width,
            height=image.height,
            pixel_type=RasterPixelType.GRAY8,
            data=bytes(pixels),
        )

    def to_raster_image(self) -> "RasterImage":
        if self.pixel_type is not RasterPixelType.GRAY8:
            raise ValueError("只有 gray8 RasterPlane 可转换为旧版 RasterImage")
        return RasterImage(
            width=self.width,
            height=self.height,
            pixels=list(self.data),
        )


def _dimension_value(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} 必须是非负整数")
    try:
        normalized = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{field_name} 必须是非负整数") from exc
    if normalized != value or normalized < 0:
        raise ValueError(f"{field_name} 必须是非负整数")
    return normalized


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
