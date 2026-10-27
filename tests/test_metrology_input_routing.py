"""Existing professional labels must not steal acquisition commands."""
from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QKeyEvent

from fdm.geometry import Point
from test_metrology_workspace import mounted_window, _begin, _distance, _mouse, _paint


@pytest.mark.parametrize("delta", [QPointF(0, 0), QPointF(2.9, 0), QPointF(1.9, 1.9)])
def test_stationary_or_subthreshold_label_click_never_creates_history(mounted_window, delta):
    _window, document, workspace, _plane = mounted_window()
    definition = _distance(workspace)
    controller = workspace.controller()
    _paint(controller)
    document.mark_session_saved()
    before = document.to_dict()
    count = document.history.command_count
    stamp = document.state_stamp
    start = controller.label_hits[definition.id].text_rect.center()
    controller.canvas.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, start))
    controller.canvas.mouseMoveEvent(_mouse(QEvent.Type.MouseMove, start + delta))
    controller.canvas.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease, start + delta))
    assert document.get_professional_measurement(definition.id).annotation_layout == definition.annotation_layout
    assert document.history.command_count == count
    assert document.state_stamp == stamp
    assert not document.dirty_flags.session_dirty
    assert document.to_dict() == before


def test_drag_returning_to_original_position_does_not_commit(mounted_window):
    _window, document, workspace, _plane = mounted_window()
    definition = _distance(workspace)
    controller = workspace.controller()
    _paint(controller)
    count = document.history.command_count
    start = controller.label_hits[definition.id].text_rect.center()
    controller.canvas.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, start))
    controller.canvas.mouseMoveEvent(_mouse(QEvent.Type.MouseMove, start + QPointF(30, 20)))
    assert controller.preview_layouts
    controller.canvas.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease, start))
    assert document.history.command_count == count
    assert document.get_professional_measurement(definition.id).annotation_layout == definition.annotation_layout


@pytest.mark.parametrize("zoom", [.5, 3.0])
def test_three_logical_pixel_threshold_is_independent_of_zoom_and_commits_once(mounted_window, zoom):
    window, document, workspace, _plane = mounted_window()
    definition = _distance(workspace)
    controller = workspace.controller()
    controller.canvas._zoom = zoom
    _paint(controller)
    count = document.history.command_count
    start = controller.label_hits[definition.id].text_rect.center()
    controller.canvas.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, start))
    controller.canvas.mouseMoveEvent(_mouse(QEvent.Type.MouseMove, start + QPointF(2.9, 0)))
    assert not controller.preview_layouts
    controller.canvas.mouseMoveEvent(_mouse(QEvent.Type.MouseMove, start + QPointF(3.1, 0)))
    assert controller.preview_layouts
    controller.canvas.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease, start + QPointF(4, 0)))
    assert document.history.command_count == count + 1
    window.undo_current_document()
    assert document.get_professional_measurement(definition.id).annotation_layout == definition.annotation_layout


@pytest.mark.parametrize("state", ["construction", "polygon", "line", "roi", "scale_anchor", "pan"])
def test_label_and_delete_do_not_steal_other_canvas_command(mounted_window, monkeypatch, state):
    _window, document, workspace, _plane = mounted_window()
    definition = _distance(workspace)
    controller = workspace.controller()
    canvas = controller.canvas
    _paint(controller)
    start = controller.label_hits[definition.id].text_rect.center()
    count = document.history.command_count
    with monkeypatch.context() as context:
        context.setattr(canvas, "_tool_mode", "manual")
        if state == "construction": context.setattr(canvas, "_tool_mode", "construction")
        elif state == "polygon": context.setattr(canvas, "_drawing_polygon_points", [Point(10, 10)])
        elif state == "line": context.setattr(canvas, "_drawing_line", object())
        elif state == "roi": context.setattr(canvas, "_roi_capture", object())
        elif state == "scale_anchor": context.setattr(canvas, "_scale_anchor_pick_active", True)
        else: context.setattr(canvas, "_panning", True)
        assert not controller.mouse_press(_mouse(QEvent.Type.MouseButtonPress, start))
        assert not controller.mouse_move(_mouse(QEvent.Type.MouseMove, start))
        assert controller.drag is None and controller.hovered_id is None
        for key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            assert not controller.key_press(QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier))
        assert document.get_professional_measurement(definition.id) is not None
        assert document.history.command_count == count


def test_professional_creation_uses_click_over_existing_label_as_new_control_point(mounted_window):
    _window, document, workspace, _plane = mounted_window()
    definition = _distance(workspace)
    controller = workspace.controller()
    _paint(controller)
    position = controller.label_hits[definition.id].text_rect.center()
    _begin(workspace, "distance")
    assert controller.mouse_press(_mouse(QEvent.Type.MouseButtonPress, position))
    assert len(controller.points) == 1
    assert controller.drag is None
    assert len(document.professional_measurements) == 1
    workspace.cancel()


def test_idle_selected_professional_dimension_can_still_be_deleted(mounted_window):
    _window, document, workspace, _plane = mounted_window()
    definition = _distance(workspace)
    controller = workspace.controller()
    assert controller.key_press(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Delete, Qt.KeyboardModifier.NoModifier))
    assert document.get_professional_measurement(definition.id) is None
