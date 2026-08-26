from __future__ import annotations

import os
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox, QPushButton

from fdm.services.screenshot_capture import (
    ANNOTATABLE_CAPTURE_MODES,
    INSTANT_CAPTURE_MODES,
    CaptureMode,
    CaptureRect,
    CapturedFrame,
    ScreenInfo,
    should_open_annotation,
)
from fdm.ui.screenshot_annotation_overlay import (
    CaptureViewportMapping,
    InlineAnnotationOverlay,
    screen_topology_signature,
)
from fdm.ui.screenshot_editor import (
    EditCommand,
    EditorTool,
    InlineTextEdit,
    ScreenshotEditModel,
    command_rect,
    render_edit_commands,
)


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _image(width: int = 320, height: int = 200) -> QImage:
    image = QImage(width, height, QImage.Format.Format_ARGB32)
    for y in range(height):
        for x in range(width):
            value = 255 if (x // 4 + y // 4) % 2 else 0
            image.setPixelColor(x, y, QColor(value, value, value))
    return image


def _mixed_screens() -> tuple[ScreenInfo, ...]:
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


@pytest.mark.parametrize("mode", tuple(CaptureMode))
def test_annotation_policy_has_explicit_instant_and_eligible_mode_sets(mode: CaptureMode) -> None:
    if mode in INSTANT_CAPTURE_MODES:
        assert should_open_annotation(mode, True, default=True) is False
        assert mode not in ANNOTATABLE_CAPTURE_MODES
    else:
        assert mode in ANNOTATABLE_CAPTURE_MODES
        assert should_open_annotation(mode, None, default=True) is True
        assert should_open_annotation(mode, False, default=True) is False
        assert should_open_annotation(mode, True, default=False) is True


def test_piecewise_mapping_round_trips_negative_mixed_dpi_coordinates() -> None:
    mapping = CaptureViewportMapping(
        CaptureRect(-400, 100, 800, 400),
        _mixed_screens(),
    )
    visible = CaptureRect(0, 0, 800, 400).to_qrect()

    for physical in (QPointF(-350, 140), QPointF(250, 220)):
        widget = mapping.physical_to_widget(physical)
        assert widget is not None
        restored = mapping.widget_to_physical(widget)
        assert restored is not None
        assert restored.x() == pytest.approx(physical.x(), abs=1.1)
        assert restored.y() == pytest.approx(physical.y(), abs=1.1)

    assert len(mapping.image_fragments(visible)) == 2
    assert mapping.logical_capture_rect().width > 0


@pytest.mark.parametrize("ratio", (1.0, 1.25, 1.5, 2.0))
def test_mapping_round_trip_common_windows_dpi_scales(ratio: float) -> None:
    logical = CaptureRect(0, 0, 800, 600)
    physical = CaptureRect(0, 0, round(800 * ratio), round(600 * ratio))
    screen = ScreenInfo("screen", logical, physical, ratio, True)
    mapping = CaptureViewportMapping(physical, (screen,))
    point = QPointF(337 * ratio, 241 * ratio)

    widget = mapping.physical_to_widget(point)
    assert widget is not None
    restored = mapping.widget_to_physical(widget)
    assert restored is not None
    assert restored.x() == pytest.approx(point.x(), abs=1.1)
    assert restored.y() == pytest.approx(point.y(), abs=1.1)


def test_topology_signature_detects_dpi_or_geometry_changes() -> None:
    screens = _mixed_screens()
    changed = (
        screens[0],
        ScreenInfo(
            screens[1].name,
            screens[1].logical_rect,
            screens[1].physical_rect,
            1.25,
            True,
        ),
    )
    assert screen_topology_signature(screens) != screen_topology_signature(changed)


def test_model_select_move_resize_duplicate_layer_delete_and_single_drag_undo() -> None:
    model = ScreenshotEditModel(_image(160, 120))
    bottom = EditCommand.from_drag(EditorTool.RECTANGLE, (10, 10), (80, 70), fill_color="#ff0000")
    top = EditCommand.from_drag(EditorTool.RECTANGLE, (30, 25), (100, 90), fill_color="#00ff00")
    model.add_command(bottom)
    model.add_command(top)

    assert EditCommand.from_dict(top.to_dict()).id == top.id
    assert model.select_at((40, 40)) == top
    initial = model.selection_bounds()
    assert model.move_selected(7, 9)
    moved = model.selection_bounds()
    assert moved.topLeft() == initial.topLeft() + QPointF(7, 9)
    assert model.undo()
    assert model.selection_bounds() == initial
    assert model.redo()

    assert model.resize_selected(QRectF(20, 20, 100, 80))
    duplicates = model.duplicate_selected()
    assert len(duplicates) == 1 and duplicates[0] != top.id
    assert model.send_to_back()
    assert model.commands[0].id == duplicates[0]
    assert model.bring_to_front()
    assert model.commands[-1].id == duplicates[0]
    assert model.delete_selected()
    assert all(item.id != duplicates[0] for item in model.commands)


def test_model_multi_select_line_endpoint_and_bulk_properties() -> None:
    model = ScreenshotEditModel(_image(180, 120))
    line = EditCommand(EditorTool.LINE, points=((5, 5), (80, 40)))
    arrow = EditCommand(EditorTool.ARROW, points=((20, 80), (120, 30)))
    model.add_command(line)
    model.add_command(arrow)
    model.set_selection((line.id, arrow.id))

    assert model.update_selected(color="#1565c0", stroke_width=7)
    assert all(item.color == "#1565c0" and item.stroke_width == 7 for item in model.selected_commands)
    model.set_selection((line.id,))
    assert model.set_line_endpoint(line.id, -1, (140, 90))
    updated = next(item for item in model.commands if item.id == line.id)
    assert updated.points[-1] == (140.0, 90.0)


def test_inline_selection_previews_endpoint_and_resize_without_committing() -> None:
    _app()
    screen = ScreenInfo(
        "preview",
        CaptureRect(0, 0, 500, 360),
        CaptureRect(0, 0, 500, 360),
        1.0,
        True,
    )
    frame = CapturedFrame(
        _image(240, 160),
        CaptureRect(100, 80, 240, 160),
        CaptureMode.REGION,
    )
    overlay = InlineAnnotationOverlay(frame, (screen,))
    line = EditCommand(EditorTool.ARROW, points=((10, 10), (60, 20)))
    rectangle = EditCommand.from_drag(EditorTool.RECTANGLE, (20, 30), (90, 80))
    try:
        overlay.model.add_command(line)
        overlay.set_tool(EditorTool.SELECT)
        overlay.model.set_selection((line.id,))
        baseline = overlay.model.commands
        overlay._drag_origin = (60, 20)
        overlay._drag_current = (110, 95)
        overlay._line_endpoint = (line.id, -1)

        bounds, previews = overlay._selection_preview()

        assert previews[0].points == ((10.0, 10.0), (110.0, 95.0))
        assert bounds.contains(QPointF(110, 95))
        assert overlay.model.commands == baseline

        overlay._drag_origin = None
        overlay._drag_current = None
        overlay._line_endpoint = None
        overlay.model.add_command(rectangle)
        overlay.model.set_selection((rectangle.id,))
        overlay._drag_origin = (90, 80)
        overlay._drag_current = (130, 120)
        overlay._resize_handle = "se"
        bounds, previews = overlay._selection_preview()

        assert bounds.bottomRight() == QPointF(130, 120)
        assert command_rect(previews[0]).bottomRight() == QPointF(130, 120)
    finally:
        overlay.close()


def test_true_blur_differs_from_mosaic_and_crop_never_expands() -> None:
    base = _image(120, 90)
    region = (10, 10, 80, 60)
    mosaic = render_edit_commands(base, (EditCommand(EditorTool.MOSAIC, rect=region, block_size=12),))
    blurred = render_edit_commands(base, (EditCommand(EditorTool.BLUR, rect=region, block_size=12),))

    assert blurred != mosaic
    assert 0 < blurred.pixelColor(40, 40).red() < 255

    model = ScreenshotEditModel(base)
    assert model.set_crop((-100, -100, 500, 500))
    assert model.visible_rect == base.rect()
    assert model.set_crop((10, 8, 50, 40))
    assert model.render().size().toTuple() == (50, 40)
    assert model.set_crop((-50, -50, 500, 500))
    assert model.render().size().toTuple() == (50, 40)


def test_inline_text_accepts_chinese_multiline_and_enter_hierarchy() -> None:
    _app()
    editor = InlineTextEdit()
    submitted: list[bool] = []
    editor.submitted.connect(lambda: submitted.append(True))
    try:
        editor.show()
        editor.setPlainText("中文输入")
        QTest.keyClick(editor, Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier)
        QTest.keyClicks(editor, "second")
        assert "\nsecond" in editor.toPlainText()
        QTest.keyClick(editor, Qt.Key.Key_Return)
        assert submitted == [True]
    finally:
        editor.close()


def test_inline_text_reedit_preserves_style_and_edge_placement_commits() -> None:
    app = _app()
    screen = ScreenInfo(
        "primary",
        CaptureRect(0, 0, 420, 300),
        CaptureRect(0, 0, 420, 300),
        1.0,
        True,
    )
    frame = CapturedFrame(
        _image(120, 80),
        CaptureRect(150, 100, 120, 80),
        CaptureMode.REGION,
    )
    overlay = InlineAnnotationOverlay(frame, (screen,))
    original = EditCommand(
        EditorTool.TEXT,
        points=((10, 30),),
        rect=(8, 8, 90, 42),
        text="原文",
        color="#1565c0",
        opacity=0.7,
        font_family="Arial",
        font_size=24,
        bold=True,
        italic=True,
        background_color="#80202020",
    )
    try:
        overlay.begin()
        app.processEvents()
        overlay.model.add_command(original)
        overlay._tool_style(EditorTool.TEXT).update(
            color="#ff0000",
            font_size=10,
            background_color="",
        )
        overlay._begin_text_edit(QPointF(10, 10), command=original)
        assert overlay._text_edit is not None
        overlay._text_edit.setPlainText("新文\n第二行")
        overlay._commit_text_edit()

        updated = next(item for item in overlay.model.commands if item.id == original.id)
        assert updated.text == "新文\n第二行"
        assert (
            updated.color,
            updated.opacity,
            updated.font_family,
            updated.font_size,
            updated.bold,
            updated.italic,
            updated.background_color,
        ) == (
            original.color,
            original.opacity,
            original.font_family,
            original.font_size,
            original.bold,
            original.italic,
            original.background_color,
        )

        overlay.set_tool(EditorTool.TEXT)
        overlay._begin_text_edit(QPointF(118, 78))
        assert overlay._text_edit is not None
        assert overlay._display_capture_bounds().toAlignedRect().contains(
            overlay._text_edit.geometry()
        )
        overlay._text_edit.setPlainText("边缘文字")
        overlay._commit_text_edit()
        assert overlay._text_edit is None
        assert overlay.model.commands[-1].text == "边缘文字"
    finally:
        overlay.close()


def test_drawing_constraints_snap_angle_square_and_center_origin() -> None:
    _app()
    screen = ScreenInfo(
        "primary",
        CaptureRect(0, 0, 640, 480),
        CaptureRect(0, 0, 640, 480),
        1.0,
        True,
    )
    frame = CapturedFrame(_image(240, 160), CaptureRect(100, 80, 240, 160), CaptureMode.REGION)
    overlay = InlineAnnotationOverlay(frame, (screen,))
    try:
        overlay.set_tool(EditorTool.LINE)
        line = overlay._command_from_points(
            ((20, 20), (100, 55)), Qt.KeyboardModifier.ShiftModifier
        )
        assert line is not None
        dx = line.points[-1][0] - line.points[0][0]
        dy = line.points[-1][1] - line.points[0][1]
        assert abs(dx) == pytest.approx(abs(dy))

        overlay.set_tool(EditorTool.RECTANGLE)
        square = overlay._command_from_points(
            ((80, 70), (120, 90)), Qt.KeyboardModifier.ShiftModifier
        )
        centered = overlay._command_from_points(
            ((80, 70), (120, 90)), Qt.KeyboardModifier.ControlModifier
        )
        assert square is not None and square.rect[2] == square.rect[3]
        assert centered is not None and centered.rect == (40.0, 50.0, 80.0, 40.0)
    finally:
        overlay.close()


def test_auto_number_continues_within_session_without_overwriting_saved_start() -> None:
    app = _app()
    screen = ScreenInfo(
        "primary",
        CaptureRect(0, 0, 640, 480),
        CaptureRect(0, 0, 640, 480),
        1.0,
        True,
    )
    frame = CapturedFrame(
        _image(240, 160),
        CaptureRect(100, 80, 240, 160),
        CaptureMode.REGION,
    )
    overlay = InlineAnnotationOverlay(frame, (screen,))
    try:
        overlay.begin()
        overlay.set_tool(EditorTool.NUMBER)
        overlay.number_spin.setValue(5)
        first = overlay._image_to_widget(QPointF(30, 30))
        assert first is not None
        QTest.mouseClick(overlay, Qt.MouseButton.LeftButton, pos=first.toPoint())
        assert overlay.model.commands[-1].number == 5
        assert overlay.number_spin.value() == 6

        overlay.set_tool(EditorTool.RECTANGLE)
        overlay.set_tool(EditorTool.NUMBER)
        assert overlay.number_spin.value() == 6
        second = overlay._image_to_widget(QPointF(80, 60))
        assert second is not None
        QTest.mouseClick(overlay, Qt.MouseButton.LeftButton, pos=second.toPoint())

        assert overlay.model.commands[-1].number == 6
        assert overlay._tool_style(EditorTool.NUMBER)["number_start"] == 5
        app.processEvents()
    finally:
        overlay.close()


def test_overlay_small_selection_keeps_finish_visible_and_uses_more_menu() -> None:
    app = _app()
    screen = ScreenInfo(
        "primary",
        CaptureRect(0, 0, 640, 360),
        CaptureRect(0, 0, 640, 360),
        1.0,
        True,
    )
    frame = CapturedFrame(_image(120, 80), CaptureRect(260, 130, 120, 80), CaptureMode.REGION)
    overlay = InlineAnnotationOverlay(frame, (screen,))
    try:
        overlay.begin()
        app.processEvents()
        assert overlay.more_button.isVisible()
        finish = overlay.toolbar.findChild(QPushButton, "finishButton")
        assert finish is not None and finish.isVisible()
        assert overlay.rect().contains(overlay.toolbar.geometry())
        assert overlay.rect().contains(overlay.properties.geometry())
        assert not overlay.toolbar.geometry().intersects(overlay._display_capture_bounds().toAlignedRect())
        assert not overlay.properties.geometry().intersects(overlay._display_capture_bounds().toAlignedRect())
        assert not overlay.undo_button.isEnabled()
        overlay.model.add_command(
            EditCommand.from_drag(EditorTool.RECTANGLE, (5, 5), (40, 30))
        )
        assert overlay.undo_button.isEnabled()
        overlay.model.undo()
        assert not overlay.undo_button.isEnabled()
        assert overlay._compact_redo_action is not None
        assert overlay._compact_redo_action.isEnabled()
    finally:
        overlay.close()


def test_zoomed_small_desktop_keeps_low_frequency_tools_in_more_menu() -> None:
    app = _app()
    screen = ScreenInfo(
        "primary",
        CaptureRect(0, 0, 640, 480),
        CaptureRect(0, 0, 640, 480),
        1.0,
        True,
    )
    frame = CapturedFrame(
        _image(560, 360),
        CaptureRect(40, 40, 560, 360),
        CaptureMode.REGION,
    )
    overlay = InlineAnnotationOverlay(frame, (screen,))
    try:
        overlay.begin()
        overlay.set_zoom(2.0)
        app.processEvents()

        assert overlay.more_button.isVisible()
        assert not overlay._tool_buttons[EditorTool.ELLIPSE].isVisible()
        assert overlay.finish_button.isVisible()
        assert overlay.rect().contains(overlay.toolbar.geometry())
    finally:
        overlay.close()


def test_context_properties_hide_irrelevant_controls_and_keep_tool_guidance() -> None:
    _app()
    screen = ScreenInfo(
        "primary",
        CaptureRect(0, 0, 800, 600),
        CaptureRect(0, 0, 800, 600),
        1.0,
        True,
    )
    frame = CapturedFrame(
        _image(320, 200),
        CaptureRect(180, 140, 320, 200),
        CaptureMode.REGION,
    )
    overlay = InlineAnnotationOverlay(frame, (screen,))
    try:
        overlay.set_tool(EditorTool.SELECT)
        assert overlay.selection_hint.isVisibleTo(overlay.properties)
        assert not overlay.color_button.isVisibleTo(overlay.properties)

        overlay.set_tool(EditorTool.BLUR)
        assert overlay.strength_spin.isVisibleTo(overlay.properties)
        assert not overlay.color_button.isVisibleTo(overlay.properties)
        assert not overlay.width_spin.isVisibleTo(overlay.properties)

        overlay.set_tool(EditorTool.TEXT)
        assert overlay.font_combo.isVisibleTo(overlay.properties)
        assert overlay.fill_button.isVisibleTo(overlay.properties)
        assert not overlay.strength_spin.isVisibleTo(overlay.properties)

        overlay.set_tool(EditorTool.CROP)
        assert overlay.crop_hint.isVisibleTo(overlay.properties)
        assert not overlay.color_button.isVisibleTo(overlay.properties)
        assert not overlay.opacity_slider.isVisibleTo(overlay.properties)
    finally:
        overlay.close()


def test_floating_controls_are_clamped_to_capture_screen() -> None:
    app = _app()
    screens = (
        ScreenInfo(
            "left-review",
            CaptureRect(-640, 0, 640, 480),
            CaptureRect(-640, 0, 640, 480),
            1.0,
        ),
        ScreenInfo(
            "right-review",
            CaptureRect(0, 0, 640, 480),
            CaptureRect(0, 0, 640, 480),
            1.0,
            True,
        ),
    )
    frame = CapturedFrame(
        _image(220, 140),
        CaptureRect(210, 120, 220, 140),
        CaptureMode.REGION,
    )
    overlay = InlineAnnotationOverlay(frame, screens)
    try:
        overlay.begin()
        app.processEvents()
        right_screen_local = QRectF(640, 0, 640, 480).toAlignedRect()

        assert right_screen_local.contains(overlay.toolbar.geometry())
        assert right_screen_local.contains(overlay.properties.geometry())
    finally:
        overlay.close()


def test_4k_draft_repaints_reuse_committed_render_cache() -> None:
    app = _app()
    image = QImage(3840, 2160, QImage.Format.Format_ARGB32)
    image.fill(QColor("#334455"))
    screen = ScreenInfo(
        "4k",
        CaptureRect(0, 0, 1920, 1080),
        CaptureRect(0, 0, 3840, 2160),
        2.0,
        True,
    )
    frame = CapturedFrame(image, screen.physical_rect, CaptureMode.FULL_SCREEN)
    overlay = InlineAnnotationOverlay(frame, (screen,))
    try:
        overlay.begin()
        app.processEvents()
        baseline = overlay.model.render_count
        overlay.set_tool(EditorTool.PEN)
        overlay._points = [(100, 100), (200, 160), (300, 180)]
        for _index in range(5):
            overlay.update()
            app.processEvents()
        assert overlay.model.render_count == baseline
    finally:
        overlay.close()


def test_topology_change_requests_fallback_without_losing_commands() -> None:
    _app()
    screen = ScreenInfo(
        "primary",
        CaptureRect(0, 0, 800, 600),
        CaptureRect(0, 0, 800, 600),
        1.0,
        True,
    )
    changed = ScreenInfo(
        "primary",
        CaptureRect(0, 0, 640, 480),
        CaptureRect(0, 0, 800, 600),
        1.25,
        True,
    )
    frame = CapturedFrame(_image(200, 120), CaptureRect(100, 80, 200, 120), CaptureMode.REGION)
    overlay = InlineAnnotationOverlay(frame, (screen,), screens_provider=lambda: (changed,))
    fallback: list[ScreenshotEditModel] = []
    overlay.fallbackRequested.connect(fallback.append)
    command = EditCommand.from_drag(EditorTool.RECTANGLE, (10, 10), (80, 60))
    overlay.model.add_command(command)
    try:
        overlay._check_topology()

        assert fallback == [overlay.model]
        assert fallback[0].parent() is None
        assert fallback[0].commands[0].id == command.id
    finally:
        overlay.close()


def test_output_pending_blocks_duplicate_output_and_topology_handoff() -> None:
    _app()
    screen = ScreenInfo(
        "primary",
        CaptureRect(0, 0, 800, 600),
        CaptureRect(0, 0, 800, 600),
        1.0,
        True,
    )
    changed = ScreenInfo(
        "primary",
        CaptureRect(0, 0, 640, 480),
        CaptureRect(0, 0, 800, 600),
        1.25,
        True,
    )
    frame = CapturedFrame(
        _image(200, 120),
        CaptureRect(100, 80, 200, 120),
        CaptureMode.REGION,
    )
    overlay = InlineAnnotationOverlay(
        frame,
        (screen,),
        screens_provider=lambda: (changed,),
    )
    copies: list[QImage] = []
    fallback: list[ScreenshotEditModel] = []
    overlay.copyRequested.connect(copies.append)
    overlay.fallbackRequested.connect(fallback.append)
    try:
        overlay.request_copy()
        overlay.request_copy()
        overlay._check_topology()

        assert len(copies) == 1
        assert overlay.output_pending
        assert fallback == []
    finally:
        overlay.close()


def test_escape_and_right_click_cancel_in_layers_without_silent_data_loss() -> None:
    app = _app()
    screen = ScreenInfo(
        "primary",
        CaptureRect(0, 0, 640, 480),
        CaptureRect(0, 0, 640, 480),
        1.0,
        True,
    )
    frame = CapturedFrame(_image(240, 160), CaptureRect(180, 120, 240, 160), CaptureMode.REGION)
    overlay = InlineAnnotationOverlay(frame, (screen,))
    cancelled: list[bool] = []
    overlay.cancelled.connect(lambda: cancelled.append(True))
    try:
        overlay.begin()
        app.processEvents()
        overlay._points = [(5, 5), (30, 20)]
        QTest.keyClick(overlay, Qt.Key.Key_Escape)
        assert overlay._points == [] and cancelled == []

        command = EditCommand.from_drag(EditorTool.RECTANGLE, (10, 10), (80, 60))
        overlay.model.add_command(command)
        overlay.set_tool(EditorTool.SELECT)
        overlay.model.set_selection((command.id,))
        QTest.keyClick(overlay, Qt.Key.Key_Escape)
        assert overlay.model.selected_ids == () and cancelled == []

        overlay._points = [(8, 8), (40, 30)]
        QTest.mouseClick(overlay, Qt.MouseButton.RightButton, pos=QPoint(200, 150))
        assert overlay._points == [] and cancelled == []

        with patch(
            "fdm.ui.screenshot_annotation_overlay.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Yes,
        ):
            QTest.keyClick(overlay, Qt.Key.Key_Escape)
        assert cancelled == [True]
    finally:
        overlay.close()
