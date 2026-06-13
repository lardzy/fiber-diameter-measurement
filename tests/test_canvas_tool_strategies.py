from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm.geometry import Line, Point
from fdm.ui.canvas_tool_strategies import (
    ContinuousManualToolStrategy,
    CountToolStrategy,
    LineToolStrategy,
    clamp_point_to_image,
)


class LineToolStrategyTests(unittest.TestCase):
    def test_begin_preview_and_cancel_manage_line_state(self) -> None:
        strategy = LineToolStrategy()
        anchor = Point(10, 12)

        state = strategy.begin(anchor, commit_on_second_click=True)

        self.assertEqual(state.anchor_raw, anchor)
        self.assertEqual(state.preview_line, Line(anchor, anchor))
        self.assertTrue(state.commit_on_second_click)
        self.assertIsNone(strategy.cancel().preview_line)

    def test_commit_payload_requires_one_pixel_line_length(self) -> None:
        strategy = LineToolStrategy()

        self.assertIsNone(strategy.commit_payload(Line(Point(4, 4), Point(4.5, 4))))
        line = Line(Point(4, 4), Point(6, 4))

        self.assertEqual(strategy.commit_payload(line), line)

    def test_shift_constraint_uses_dominant_axis_for_preview_line(self) -> None:
        strategy = LineToolStrategy()

        horizontal = strategy.preview_line(
            Point(10, 20),
            Point(60, 35),
            image_size=(120, 90),
            constrain_axis=True,
            snap_to_pixel=False,
            snap_anchor=True,
        )
        vertical = strategy.preview_line(
            Point(10, 20),
            Point(22, 75),
            image_size=(120, 90),
            constrain_axis=True,
            snap_to_pixel=False,
            snap_anchor=True,
        )

        self.assertEqual(horizontal.end, Point(60, 20))
        self.assertEqual(vertical.end, Point(10, 75))

    def test_ctrl_snap_uses_pixel_centers_and_clamps_to_image(self) -> None:
        strategy = LineToolStrategy()

        line = strategy.preview_line(
            Point(-2.2, 20.7),
            Point(60.9, 99.2),
            image_size=(50, 80),
            constrain_axis=False,
            snap_to_pixel=True,
            snap_anchor=True,
        )

        self.assertEqual(line.start, Point(0.5, 20.5))
        self.assertEqual(line.end, Point(49.5, 79.5))

    def test_dragging_existing_endpoint_does_not_snap_fixed_anchor(self) -> None:
        strategy = LineToolStrategy()

        line = strategy.preview_line(
            Point(10.2, 20.7),
            Point(30.9, 26.1),
            image_size=(80, 80),
            constrain_axis=False,
            snap_to_pixel=True,
            snap_anchor=False,
        )

        self.assertEqual(line.start, Point(10.2, 20.7))
        self.assertEqual(line.end, Point(30.5, 26.5))

    def test_anchor_for_event_applies_ctrl_snap(self) -> None:
        strategy = LineToolStrategy()

        anchor = strategy.anchor_for_event(
            Point(12.9, 18.1),
            image_size=(30, 30),
            snap_to_pixel=True,
        )

        self.assertEqual(anchor, Point(12.5, 18.5))


class ContinuousManualToolStrategyTests(unittest.TestCase):
    def test_points_with_candidate_ignores_duplicate_near_last_point(self) -> None:
        strategy = ContinuousManualToolStrategy()
        points = [Point(10, 10), Point(25, 20)]

        result = strategy.points_with_candidate(points, Point(25.4, 20.2))

        self.assertEqual(result, points)

    def test_completion_candidate_preserves_strict_one_pixel_threshold(self) -> None:
        strategy = ContinuousManualToolStrategy()
        points = [Point(10, 10), Point(25, 20)]

        click_result = strategy.points_with_candidate(points, Point(26, 20))
        completion_result = strategy.points_with_candidate(
            points,
            Point(26, 20),
            include_threshold=False,
        )

        self.assertEqual(click_result, [Point(10, 10), Point(25, 20), Point(26, 20)])
        self.assertEqual(completion_result, points)

    def test_commit_payload_builds_polyline_payload(self) -> None:
        strategy = ContinuousManualToolStrategy()
        points = [Point(10, 10), Point(25, 20), Point(35, 45)]

        payload = strategy.commit_payload(points)

        self.assertIsNotNone(payload)
        self.assertEqual(payload["measurement_kind"], "polyline")
        self.assertEqual(payload["polyline_px"], points)
        self.assertIsNone(strategy.commit_payload(points[:1]))

    def test_cancel_clears_pending_polyline(self) -> None:
        self.assertEqual(ContinuousManualToolStrategy().cancel(), [])


class CountToolStrategyTests(unittest.TestCase):
    def test_count_click_builds_point_payload(self) -> None:
        point = Point(42, 58)

        payload = CountToolStrategy().commit_payload(point)

        self.assertEqual(payload["measurement_kind"], "count")
        self.assertEqual(payload["point_px"], point)


class ClampPointToImageTests(unittest.TestCase):
    def test_clamp_without_image_leaves_point_unchanged(self) -> None:
        point = Point(-3, 12)

        self.assertEqual(clamp_point_to_image(point, None, pixel_center=False), point)


if __name__ == "__main__":
    unittest.main()
