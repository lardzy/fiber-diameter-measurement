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
