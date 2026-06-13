from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from fdm.geometry import Line, Point, clamp, distance, line_length, snap_to_pixel_center


ImageSize = tuple[int, int]


class CanvasToolStrategy(Protocol):
    tool_modes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LineToolState:
    anchor_raw: Point | None = None
    preview_line: Line | None = None
    commit_on_second_click: bool = False


class LineToolStrategy:
    tool_modes = ("manual", "snap", "calibration")
    minimum_length_px = 1.0

    def begin(self, anchor: Point, *, commit_on_second_click: bool = False) -> LineToolState:
        return LineToolState(
            anchor_raw=anchor,
            preview_line=Line(start=anchor, end=anchor),
            commit_on_second_click=commit_on_second_click,
        )

    def cancel(self) -> LineToolState:
        return LineToolState()

    def preview_line(
        self,
        anchor: Point,
        candidate: Point,
        *,
        image_size: ImageSize | None,
        constrain_axis: bool,
        snap_to_pixel: bool,
        snap_anchor: bool,
    ) -> Line:
        fixed = snap_to_pixel_center(anchor) if snap_to_pixel and snap_anchor else anchor
        moving = candidate
        if constrain_axis:
            dx = moving.x - fixed.x
            dy = moving.y - fixed.y
            if abs(dx) >= abs(dy):
                moving = Point(moving.x, fixed.y)
            else:
                moving = Point(fixed.x, moving.y)
        if snap_to_pixel:
            moving = snap_to_pixel_center(moving)
        fixed = clamp_point_to_image(
            fixed,
            image_size,
            pixel_center=snap_to_pixel and snap_anchor,
        )
        moving = clamp_point_to_image(moving, image_size, pixel_center=snap_to_pixel)
        return Line(start=fixed, end=moving)

    def preview_state(
        self,
        state: LineToolState,
        candidate: Point,
        *,
        image_size: ImageSize | None,
        constrain_axis: bool,
        snap_to_pixel: bool,
        snap_anchor: bool,
    ) -> LineToolState:
        if state.anchor_raw is None:
            return state
        return LineToolState(
            anchor_raw=state.anchor_raw,
            preview_line=self.preview_line(
                state.anchor_raw,
                candidate,
                image_size=image_size,
                constrain_axis=constrain_axis,
                snap_to_pixel=snap_to_pixel,
                snap_anchor=snap_anchor,
            ),
            commit_on_second_click=state.commit_on_second_click,
        )

    def anchor_for_event(
        self,
        image_point: Point,
        *,
        image_size: ImageSize | None,
        snap_to_pixel: bool,
    ) -> Point:
        candidate = snap_to_pixel_center(image_point) if snap_to_pixel else image_point
        return clamp_point_to_image(candidate, image_size, pixel_center=snap_to_pixel)

    def can_commit(self, line: Line | None) -> bool:
        return line is not None and line_length(line) >= self.minimum_length_px

    def commit_payload(self, line: Line | None) -> Line | None:
        if not self.can_commit(line):
            return None
        return line


class ContinuousManualToolStrategy:
    tool_modes = ("continuous_manual",)
    minimum_points = 2
    duplicate_point_distance_px = 1.0

    def should_append_point(
        self,
        points: Sequence[Point],
        point: Point,
        *,
        include_threshold: bool = True,
    ) -> bool:
        if not points:
            return True
        gap = distance(points[-1], point)
        if include_threshold:
            return gap >= self.duplicate_point_distance_px
        return gap > self.duplicate_point_distance_px

    def points_with_candidate(
        self,
        points: Sequence[Point],
        point: Point,
        *,
        include_threshold: bool = True,
    ) -> list[Point]:
        result = list(points)
        if self.should_append_point(result, point, include_threshold=include_threshold):
            result.append(point)
        return result

    def cancel(self) -> list[Point]:
        return []

    def can_commit(self, points: Sequence[Point]) -> bool:
        return len(points) >= self.minimum_points

    def commit_payload(self, points: Sequence[Point]) -> dict[str, object] | None:
        if not self.can_commit(points):
            return None
        return {
            "measurement_kind": "polyline",
            "polyline_px": list(points),
        }


class CountToolStrategy:
    tool_modes = ("count",)

    def commit_payload(self, point: Point) -> dict[str, object]:
        return {
            "measurement_kind": "count",
            "point_px": point,
        }


def clamp_point_to_image(point: Point, image_size: ImageSize | None, *, pixel_center: bool) -> Point:
    if image_size is None:
        return point
    width, height = image_size
    minimum = 0.5 if pixel_center else 0.0
    maximum_x = (width - 0.5) if pixel_center else (width - 1.0)
    maximum_y = (height - 0.5) if pixel_center else (height - 1.0)
    return Point(
        x=clamp(point.x, minimum, max(minimum, maximum_x)),
        y=clamp(point.y, minimum, max(minimum, maximum_y)),
    )
