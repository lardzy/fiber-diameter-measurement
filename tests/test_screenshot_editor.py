from __future__ import annotations

import os
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QImage, QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from fdm.ui.screenshot_editor import (
    EditCommand,
    EditorTool,
    ScreenshotEditModel,
    ScreenshotEditor,
)


def _image(width: int = 40, height: int = 30) -> QImage:
    image = QImage(width, height, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.white)
    return image


def test_editor_tool_enum_instances_are_accepted() -> None:
    command = EditCommand(EditorTool.RECTANGLE, rect=(1, 2, 3, 4))

    assert command.tool is EditorTool.RECTANGLE
    assert EditorTool.parse(EditorTool.TEXT) is EditorTool.TEXT


def test_annotation_after_crop_uses_visible_canvas_coordinates() -> None:
    model = ScreenshotEditModel(_image())
    model.add_command(EditCommand(EditorTool.CROP, rect=(10, 8, 12, 10)))
    model.add_command(
        EditCommand(
            EditorTool.RECTANGLE,
            rect=(1, 1, 7, 6),
            color="#ff0000",
            stroke_width=2,
        )
    )

    rendered = model.render()

    assert (rendered.width(), rendered.height()) == (12, 10)
    assert any(
        rendered.pixelColor(x, y).red() > 220
        and rendered.pixelColor(x, y).green() < 180
        for y in range(rendered.height())
        for x in range(rendered.width())
    )


def test_empty_crop_is_ignored_and_does_not_offset_later_annotations() -> None:
    model = ScreenshotEditModel(_image())

    model.add_command(EditCommand(EditorTool.CROP, rect=(10, 8, 0, 0)))
    model.add_command(
        EditCommand(
            EditorTool.RECTANGLE,
            rect=(1, 1, 7, 6),
            color="#ff0000",
            stroke_width=2,
        )
    )

    assert all(command.tool is not EditorTool.CROP for command in model.commands)
    assert model.commands[0].rect == (1.0, 1.0, 7.0, 6.0)
    assert model.render().pixelColor(1, 1).red() > 220


def test_crop_is_clipped_before_translating_annotations_and_nested_crop() -> None:
    model = ScreenshotEditModel(_image())
    model.add_command(EditCommand(EditorTool.CROP, rect=(-10, -8, 30, 25)))
    assert model.commands[-1].rect == (0.0, 0.0, 20.0, 17.0)

    model.add_command(EditCommand(EditorTool.CROP, rect=(2, 3, 8, 7)))
    assert model.commands[-1].rect == (2.0, 3.0, 8.0, 7.0)
    model.add_command(
        EditCommand(
            EditorTool.RECTANGLE,
            rect=(1, 1, 5, 4),
            color="#ff0000",
            stroke_width=2,
        )
    )

    rendered = model.render()
    assert (rendered.width(), rendered.height()) == (8, 7)
    assert any(
        rendered.pixelColor(x, y).red() > 220
        and rendered.pixelColor(x, y).green() < 180
        for y in range(rendered.height())
        for x in range(rendered.width())
    )


def test_crop_drag_has_visible_selection_preview() -> None:
    _app = QApplication.instance() or QApplication([])
    editor = ScreenshotEditor(_image())
    editor.set_tool(EditorTool.CROP)
    try:
        editor.canvas.mousePressEvent(
            QMouseEvent(
                QEvent.Type.MouseButtonPress,
                QPointF(2, 2),
                QPointF(2, 2),
                QPointF(2, 2),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
        )
        editor.canvas.mouseMoveEvent(
            QMouseEvent(
                QEvent.Type.MouseMove,
                QPointF(20, 15),
                QPointF(20, 15),
                QPointF(20, 15),
                Qt.MouseButton.NoButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
        )
        preview = QImage(
            editor.canvas.size(),
            QImage.Format.Format_ARGB32_Premultiplied,
        )
        preview.fill(Qt.GlobalColor.transparent)
        editor.canvas.render(preview)

        assert preview.pixelColor(2, 2) != QColor(Qt.GlobalColor.white)
    finally:
        editor.close()


def test_editor_exposes_text_color_width_and_completion() -> None:
    _app = QApplication.instance() or QApplication([])
    editor = ScreenshotEditor(_image())
    completed: list[QImage] = []
    editor.completed.connect(completed.append)
    try:
        with patch(
            "fdm.ui.screenshot_editor.QInputDialog.getText",
            return_value=("检验文字", True),
        ):
            editor.set_tool(EditorTool.TEXT)
        assert editor.canvas.pending_text == "检验文字"
        assert editor.canvas.tool is EditorTool.TEXT

        with patch(
            "fdm.ui.screenshot_editor.QColorDialog.getColor",
            return_value=QColor("#1565c0"),
        ):
            editor._choose_color()
        editor.width_spin.setValue(7)

        QTest.mouseClick(
            editor.canvas,
            Qt.MouseButton.LeftButton,
            pos=QPoint(5, 6),
        )
        command = editor.model.commands[-1]
        assert command.text == "检验文字"
        assert command.color == "#1565c0"
        assert command.stroke_width == 7

        editor.complete()
        assert len(completed) == 1
        assert not completed[0].isNull()
    finally:
        editor.close()
