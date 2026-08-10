from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from fdm.services.screenshot_capture import CaptureRect, ScreenInfo, WindowCandidate
from fdm.ui.screenshot_editor import (
    EditCommand,
    EditorTool,
    ScreenshotEditModel,
    ScreenshotEditor,
    render_edit_commands,
)
from fdm.ui.screenshot_overlay import (
    ScreenshotOverlay,
    logical_point_to_physical,
    logical_rect_to_physical,
    physical_rect_to_logical,
)


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _base_image() -> QImage:
    image = QImage(120, 90, QImage.Format.Format_ARGB32)
    image.fill(QColor("white"))
    return image


def _screens() -> tuple[ScreenInfo, ...]:
    return (
        ScreenInfo(
            "left",
            CaptureRect(-1280, 0, 1280, 720),
            CaptureRect(-2560, 0, 2560, 1440),
            2.0,
        ),
        ScreenInfo(
            "primary",
            CaptureRect(0, 0, 1920, 1080),
            CaptureRect(0, 0, 1920, 1080),
            1.0,
            True,
        ),
    )


def test_overlay_coordinate_helpers_scale_local_offsets_not_global_origins() -> None:
    screens = _screens()

    assert logical_point_to_physical(QPoint(-640, 360), screens) == QPoint(-1280, 720)
    assert logical_rect_to_physical(
        CaptureRect(-1000, 100, 200, 100), screens
    ) == CaptureRect(-2000, 200, 400, 200)
    assert physical_rect_to_logical(
        CaptureRect(-2000, 200, 400, 200), screens
    ) == CaptureRect(-1000, 100, 200, 100)


def test_overlay_cycles_nested_candidates_and_accepts_without_drag() -> None:
    _app()
    screen = ScreenInfo(
        "primary",
        CaptureRect(0, 0, 800, 600),
        CaptureRect(0, 0, 800, 600),
        1.0,
        True,
    )
    parent = WindowCandidate(1, CaptureRect(50, 50, 500, 400), depth=0)
    child = WindowCandidate(2, CaptureRect(100, 100, 200, 120), parent_handle=1, depth=1)
    overlay = ScreenshotOverlay((screen,), (parent, child))
    accepted: list[object] = []
    overlay.selectionAccepted.connect(accepted.append)
    try:
        overlay._refresh_hover(QPoint(150, 150))
        assert overlay.selected_candidate == child
        assert overlay.cycle_candidate(1) == parent
        assert overlay.accept_candidate() is True
        assert accepted[0].candidate == parent
        assert accepted[0].rect == parent.rect
    finally:
        overlay.close()


def test_overlay_real_mouse_click_accepts_nested_window_candidate() -> None:
    _app()
    screen = ScreenInfo(
        "primary",
        CaptureRect(0, 0, 800, 600),
        CaptureRect(0, 0, 800, 600),
        1.0,
        True,
    )
    parent = WindowCandidate(
        1,
        CaptureRect(50, 50, 500, 400),
        depth=0,
        z_order=0,
        metadata={"root_handle": 1},
    )
    child = WindowCandidate(
        2,
        CaptureRect(100, 100, 200, 120),
        parent_handle=1,
        depth=1,
        z_order=1,
        metadata={"root_handle": 1, "ancestor_handles": (1,)},
    )
    overlay = ScreenshotOverlay((screen,), (parent, child))
    accepted: list[object] = []
    overlay.selectionAccepted.connect(accepted.append)
    try:
        overlay.begin()
        QTest.qWait(10)
        click_position = QPoint(150, 150)
        QTest.mouseMove(overlay, click_position)
        assert overlay.selected_candidate == child

        QTest.mouseClick(
            overlay,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            click_position,
        )

        assert len(accepted) == 1
        assert accepted[0].candidate == child
        assert accepted[0].rect == child.capture_rect
        assert not overlay.isVisible()
    finally:
        overlay.close()


def test_overlay_candidate_cutout_stays_hit_testable_with_near_transparent_alpha() -> None:
    app = _app()
    screen = ScreenInfo(
        "primary",
        CaptureRect(0, 0, 800, 600),
        CaptureRect(0, 0, 800, 600),
        1.0,
        True,
    )
    candidate = WindowCandidate(
        1,
        CaptureRect(100, 100, 200, 120),
        z_order=0,
        metadata={"root_handle": 1},
    )
    overlay = ScreenshotOverlay((screen,), (candidate,))
    try:
        overlay.begin()
        overlay._refresh_hover(QPoint(150, 150))
        overlay.repaint()
        app.processEvents()

        rendered = overlay.grab().toImage()
        assert rendered.pixelColor(150, 150).alpha() == 1
        assert rendered.pixelColor(700, 500).alpha() == 92
    finally:
        overlay.close()


def test_overlay_without_candidates_drags_a_free_region_from_any_interior_point() -> None:
    _app()
    screen = ScreenInfo(
        "primary",
        CaptureRect(0, 0, 800, 600),
        CaptureRect(0, 0, 800, 600),
        1.0,
        True,
    )
    overlay = ScreenshotOverlay((screen,), ())
    accepted: list[object] = []
    overlay.selectionAccepted.connect(accepted.append)
    try:
        overlay.begin()
        QTest.mousePress(
            overlay,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            QPoint(160, 130),
        )
        QTest.mouseMove(overlay, QPoint(570, 390))
        QTest.mouseRelease(
            overlay,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            QPoint(570, 390),
        )

        assert len(accepted) == 1
        assert accepted[0].candidate is None
        assert accepted[0].rect == CaptureRect(160, 130, 410, 260)
    finally:
        overlay.close()


def test_overlay_begin_primes_candidate_at_stationary_cursor() -> None:
    _app()
    screen = ScreenInfo(
        "primary",
        CaptureRect(0, 0, 800, 600),
        CaptureRect(0, 0, 800, 600),
        1.0,
        True,
    )
    overlay = ScreenshotOverlay((screen,))
    try:
        with patch.object(overlay, "_refresh_hover") as refresh:
            overlay.begin()
        refresh.assert_called_once()
    finally:
        overlay.close()


def test_editor_command_round_trip_history_crop_and_render_all_tools() -> None:
    _app()
    model = ScreenshotEditModel(_base_image())
    rectangle = EditCommand.from_drag(
        EditorTool.RECTANGLE,
        (5, 5),
        (60, 40),
        color="#ff0000",
    )
    assert EditCommand.from_dict(rectangle.to_dict()) == rectangle

    model.add_command(rectangle)
    rendered = model.render()
    assert QColor(rendered.pixel(5, 5)).red() > 150
    assert model.can_undo and not model.can_redo
    assert model.undo() and not model.can_undo and model.can_redo
    assert model.render().pixelColor(5, 5) == QColor("white")
    assert model.redo() and model.can_undo

    commands = (
        EditCommand.from_drag(EditorTool.ELLIPSE, (10, 10), (50, 40)),
        EditCommand(EditorTool.ARROW, points=((5, 70), (70, 50))),
        EditCommand(EditorTool.LINE, points=((0, 0), (119, 89))),
        EditCommand(EditorTool.PEN, points=((5, 5), (10, 8), (20, 7))),
        EditCommand(EditorTool.TEXT, points=((10, 30),), text="测试"),
        EditCommand(EditorTool.NUMBER, points=((90, 20),), number=3),
        EditCommand.from_drag(
            EditorTool.HIGHLIGHT,
            (20, 50),
            (90, 70),
            color="#fff176",
            opacity=0.4,
        ),
        EditCommand.from_drag(EditorTool.MOSAIC, (70, 45), (110, 80)),
        EditCommand.from_drag(EditorTool.BLUR, (0, 40), (30, 80)),
        EditCommand(EditorTool.CROP, rect=(10, 10, 80, 60)),
    )
    output = render_edit_commands(_base_image(), commands)
    assert not output.isNull()
    assert output.size().width() == 80 and output.size().height() == 60


def test_editor_save_and_clipboard_use_rendered_image(tmp_path: Path) -> None:
    _app()
    editor = ScreenshotEditor(_base_image())
    editor.add_command(
        EditCommand.from_drag(
            EditorTool.RECTANGLE,
            (4, 4),
            (50, 30),
            color="#00ff00",
        )
    )
    target = tmp_path / "edited.png"
    try:
        assert editor.save(target)
        assert target.exists() and not QImage(str(target)).isNull()
        assert editor.copy_to_clipboard()
        assert not QApplication.clipboard().image().isNull()
        assert editor.undo() and editor.redo()
    finally:
        editor.close()
