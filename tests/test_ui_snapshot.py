from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm.geometry import Point
from fdm.ui_snapshot import (
    UI_SNAPSHOT_SCENARIOS,
    _apply_measurement_fullscreen_scene,
    _apply_measurement_zoomed_scene,
)


class _CanvasStub:
    def __init__(self) -> None:
        self.zoom: float | None = None
        self.center: Point | None = None

    def set_view_zoom(self, zoom: float) -> None:
        self.zoom = zoom

    def center_on_image_point(self, point: Point) -> None:
        self.center = point


class _WindowStub:
    def __init__(self, canvas: _CanvasStub | None) -> None:
        self._canvas = canvas

    def current_canvas(self) -> _CanvasStub | None:
        return self._canvas


class _FullscreenControllerStub:
    def __init__(self) -> None:
        self.is_active = False


class _FullscreenWindowStub:
    def __init__(self) -> None:
        self._fullscreen_controller = _FullscreenControllerStub()
        self.toggle_calls: list[bool] = []

    def _toggle_fullscreen_measurement(self, checked: bool = False) -> None:
        self.toggle_calls.append(checked)
        self._fullscreen_controller.is_active = bool(checked)


class UiSnapshotScenarioTests(unittest.TestCase):
    def test_required_review_scenarios_include_zoomed_measurement(self) -> None:
        self.assertEqual(len(UI_SNAPSHOT_SCENARIOS), len(set(UI_SNAPSHOT_SCENARIOS)))
        self.assertIn("measurement", UI_SNAPSHOT_SCENARIOS)
        self.assertIn("measurement-fullscreen", UI_SNAPSHOT_SCENARIOS)
        self.assertIn("measurement-zoomed", UI_SNAPSHOT_SCENARIOS)
        self.assertIn("measurement-results", UI_SNAPSHOT_SCENARIOS)
        self.assertIn("digital-slide", UI_SNAPSHOT_SCENARIOS)
        self.assertIn("settings", UI_SNAPSHOT_SCENARIOS)
        self.assertIn("current-image-export", UI_SNAPSHOT_SCENARIOS)
        self.assertIn("measurement-export", UI_SNAPSHOT_SCENARIOS)
        self.assertIn("image-batch", UI_SNAPSHOT_SCENARIOS)
        self.assertIn("analysis-results", UI_SNAPSHOT_SCENARIOS)
        self.assertIn("advanced-analysis", UI_SNAPSHOT_SCENARIOS)

    def test_zoomed_scene_applies_deterministic_zoom_and_center(self) -> None:
        canvas = _CanvasStub()
        window = _WindowStub(canvas)

        applied = _apply_measurement_zoomed_scene(window)  # type: ignore[arg-type]

        self.assertTrue(applied)
        self.assertEqual(canvas.zoom, 2.4)
        self.assertEqual(canvas.center, Point(760.0, 430.0))

    def test_zoomed_scene_is_safe_without_an_open_canvas(self) -> None:
        window = _WindowStub(None)

        applied = _apply_measurement_zoomed_scene(window)  # type: ignore[arg-type]

        self.assertFalse(applied)

    def test_fullscreen_scene_uses_production_toggle_path(self) -> None:
        window = _FullscreenWindowStub()

        applied = _apply_measurement_fullscreen_scene(window)  # type: ignore[arg-type]

        self.assertTrue(applied)
        self.assertEqual(window.toggle_calls, [True])
        self.assertTrue(window._fullscreen_controller.is_active)


if __name__ == "__main__":
    unittest.main()
