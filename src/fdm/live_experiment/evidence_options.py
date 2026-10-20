"""Versioned evidence policy and original-pixel crop geometry.

An absent policy is deliberately distinct from a new trial's crop default.
Legacy ``save_images`` therefore keeps its original full PNG/disabled meaning.
"""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Iterable


def normalize_evidence_options(value: dict | None) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, dict) or value.keys() - {"version", "mode", "padding", "minimum_size", "quality", "subsampling"}:
        raise ValueError("测量证据配置无效。")
    result = {"version": 1, "mode": "crop_jpeg", "padding": 64,
              "minimum_size": 256, "quality": 95, "subsampling": "4:4:4", **deepcopy(value)}
    if type(result["version"]) is not int or result["version"] != 1:
        raise ValueError("不支持的测量证据配置版本。")
    if result["mode"] not in ("disabled", "crop_jpeg", "full_png"):
        raise ValueError("测量证据模式无效。")
    for name, lower, upper in (("padding", 0, 4096), ("minimum_size", 1, 16384), ("quality", 1, 100)):
        if type(result[name]) is not int or not lower <= result[name] <= upper:
            raise ValueError("测量证据尺寸或压缩质量无效。")
    if result["subsampling"] != "4:4:4":
        raise ValueError("测量证据 JPEG 必须保留 4:4:4 色度采样。")
    return result


def effective_evidence_options(value: dict | None, *, save_images: bool = True) -> dict:
    return normalize_evidence_options(value) or normalize_evidence_options(
        {"version": 1, "mode": "full_png" if save_images else "disabled"})


def evidence_crop(size: tuple[int, int], points: Iterable, *, padding: int = 64,
                  minimum_size: int = 256) -> tuple[int, int, int, int]:
    """Return x/y/width/height without scaling or losing boundary endpoints.

    Coordinates are original image coordinates. ``width``/``height`` themselves
    are accepted as the edge of the last pixel, as in existing canvas tools.
    A crop cannot silently clamp an out-of-image measurement into validity.
    """
    width, height = size
    coordinates = []
    for point in points:
        if len(point) != 2 or any(isinstance(v, bool) for v in point):
            raise ValueError("测量证据坐标无效。")
        x, y = (float(v) for v in point)
        if not math.isfinite(x) or not math.isfinite(y) or not (0 <= x <= width and 0 <= y <= height):
            raise ValueError("测量证据坐标超出源图范围。")
        coordinates.append((x, y))
    if not coordinates:
        raise ValueError("局部测量证据缺少测量坐标。")

    def axis(values, extent):
        lower = max(0, math.floor(min(values)) - padding)
        upper = min(extent, math.floor(max(values)) + 1 + padding)
        length = min(extent, max(minimum_size, upper - lower))
        start = min(max(0, (lower + upper - length) // 2), extent - length)
        return start, length

    x, w = axis([point[0] for point in coordinates], width)
    y, h = axis([point[1] for point in coordinates], height)
    return x, y, w, h


def measurement_evidence_points(result: dict) -> list[tuple[float, float]]:
    """Include original manual input and all automatic statistical geometry.

    An automatic diameter may average multiple sections; its representative
    segment alone does not contain all of the visual measurement evidence.
    """
    points = []

    def collect(value):
        if isinstance(value, dict):
            for key in ("start", "end", "input_start", "input_end", "samples", "path", "center", "point"):
                if key in value:
                    collect(value[key])
        elif isinstance(value, (list, tuple)):
            if len(value) == 2 and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
                points.append(tuple(float(item) for item in value))
            else:
                for item in value:
                    collect(item)

    collect(result)
    return points
