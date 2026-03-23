from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(slots=True)
class Point:
    x: float
    y: float

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, payload: dict[str, float]) -> "Point":
        return cls(x=float(payload["x"]), y=float(payload["y"]))


@dataclass(slots=True)
class Line:
    start: Point
    end: Point

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {"start": self.start.to_dict(), "end": self.end.to_dict()}

    @classmethod
    def from_dict(cls, payload: dict[str, dict[str, float]]) -> "Line":
        return cls(
            start=Point.from_dict(payload["start"]),
            end=Point.from_dict(payload["end"]),
        )


def line_length(line: Line) -> float:
    return distance(line.start, line.end)


def midpoint(line: Line) -> Point:
    return Point(
        x=(line.start.x + line.end.x) / 2.0,
        y=(line.start.y + line.end.y) / 2.0,
    )


def distance(a: Point, b: Point) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def subtract(a: Point, b: Point) -> tuple[float, float]:
    return a.x - b.x, a.y - b.y


def add(point: Point, vector: tuple[float, float]) -> Point:
    return Point(point.x + vector[0], point.y + vector[1])


def scale(vector: tuple[float, float], factor: float) -> tuple[float, float]:
    return vector[0] * factor, vector[1] * factor


def dot(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def normalize(vector: tuple[float, float]) -> tuple[float, float]:
    length = math.hypot(vector[0], vector[1])
    if length == 0:
        return 1.0, 0.0
    return vector[0] / length, vector[1] / length


def direction(line: Line) -> tuple[float, float]:
    return normalize(subtract(line.end, line.start))


def normal(vector: tuple[float, float]) -> tuple[float, float]:
    return -vector[1], vector[0]


def project(point: Point, origin: Point, axis: tuple[float, float]) -> float:
    return dot(subtract(point, origin), axis)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def snap_to_pixel_center(point: Point) -> Point:
    return Point(x=math.floor(point.x) + 0.5, y=math.floor(point.y) + 0.5)


def nearest_endpoint(line: Line, point: Point) -> tuple[str, float]:
    start_distance = distance(line.start, point)
    end_distance = distance(line.end, point)
    if start_distance <= end_distance:
        return "start", start_distance
    return "end", end_distance


def polygon_area(points: list[Point]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        area += (point.x * next_point.y) - (next_point.x * point.y)
    return abs(area) / 2.0


def polygon_centroid(points: list[Point]) -> Point:
    if not points:
        return Point(0.0, 0.0)
    area_factor = 0.0
    cx = 0.0
    cy = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        cross = (point.x * next_point.y) - (next_point.x * point.y)
        area_factor += cross
        cx += (point.x + next_point.x) * cross
        cy += (point.y + next_point.y) * cross
    if abs(area_factor) < 1e-9:
        return polygon_bounds_center(points)
    return Point(cx / (3.0 * area_factor), cy / (3.0 * area_factor))


def polygon_bounds(points: list[Point]) -> tuple[float, float, float, float]:
    if not points:
        return 0.0, 0.0, 0.0, 0.0
    xs = [point.x for point in points]
    ys = [point.y for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def polygon_bounds_center(points: list[Point]) -> Point:
    min_x, min_y, max_x, max_y = polygon_bounds(points)
    return Point((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)


def polygon_translate(points: list[Point], dx: float, dy: float) -> list[Point]:
    return [Point(point.x + dx, point.y + dy) for point in points]


def point_in_polygon(point: Point, polygon: list[Point]) -> bool:
    if len(polygon) < 3:
        return False
    inside = False
    for index, current in enumerate(polygon):
        nxt = polygon[(index + 1) % len(polygon)]
        intersects = ((current.y > point.y) != (nxt.y > point.y)) and (
            point.x < ((nxt.x - current.x) * (point.y - current.y) / ((nxt.y - current.y) or 1e-9)) + current.x
        )
        if intersects:
            inside = not inside
    return inside


def point_to_segment_distance(point: Point, start: Point, end: Point) -> float:
    vx = end.x - start.x
    vy = end.y - start.y
    length_sq = (vx * vx) + (vy * vy)
    if length_sq == 0:
        return distance(point, start)
    projection = ((point.x - start.x) * vx + (point.y - start.y) * vy) / length_sq
    projection = max(0.0, min(1.0, projection))
    closest = Point(
        x=start.x + (projection * vx),
        y=start.y + (projection * vy),
    )
    return distance(point, closest)


def point_to_polygon_edge_distance(point: Point, polygon: list[Point]) -> float:
    if len(polygon) < 2:
        return float("inf")
    return min(
        point_to_segment_distance(point, polygon[index], polygon[(index + 1) % len(polygon)])
        for index in range(len(polygon))
    )
