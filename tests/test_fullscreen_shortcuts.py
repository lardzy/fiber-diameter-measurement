from time import monotonic
from unittest.mock import patch

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QLineEdit

from fdm.geometry import Point
from fdm.models import ImageDocument
from fdm.services.digital_slide_store import (
    DigitalSlideManifest,
    DigitalSlideStore,
    DigitalSlideTile,
)
from fdm.ui.main_window import MainWindow


@pytest.fixture(params=["image", "slide"])
def counting_window(request, tmp_path, desktop_application):
    window = MainWindow()
    window.resize(1100, 760)
    window.show()
    if request.param == "slide":
        path = tmp_path / "keys.fdmslide"
        store = DigitalSlideStore.create(
            path,
            DigitalSlideManifest(
                version=1,
                width=1600,
                height=1200,
                viewport_width=320,
                viewport_height=240,
                focus_levels=[0],
            ),
        )
        pixels = QImage(1600, 1200, QImage.Format.Format_RGB32)
        pixels.fill(Qt.GlobalColor.white)
        store.write_tile(
            DigitalSlideTile(z_index=0, x=0, y=0, width=1600, height=1200), pixels
        )
        store.close()
        window._add_digital_slide_document_from_path(path, document=None)
        document = window.current_document()
    else:
        image = QImage(320, 240, QImage.Format.Format_RGB32)
        image.fill(Qt.GlobalColor.white)
        document = ImageDocument(id="keys", path="keys.png", image_size=(320, 240))
        document.initialize_runtime_state()
        window._mount_document(document, image, tooltip=document.path)
    first = document.ensure_default_group()
    second = document.create_group(color="#E07050", label="第二类")
    document.set_active_group(first.id)
    window._update_ui_for_current_document()
    window.set_tool_mode("count")
    window._on_canvas_line_committed(
        document.id,
        "count",
        {
            "measurement_kind": "count",
            "point_px": Point(80, 80),
        },
    )
    window._fullscreen_controller.enter()
    window.activateWindow()
    window._focus_current_canvas()
    desktop_application.processEvents()
    if request.param == "slide":
        deadline = monotonic() + 4.0
        while not window.undo_action.isEnabled() and monotonic() < deadline:
            QTest.qWait(5)
    try:
        yield window, document, first, second
    finally:
        window._fullscreen_controller.exit()
        with patch.object(window, "_confirm_close_documents", return_value=True):
            window.close()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_fullscreen_keypad_switches_category_without_reclassifying_objects(
    counting_window,
):
    window, document, first, second = counting_window
    canvas = window.current_canvas()
    QTest.keyClick(canvas, Qt.Key.Key_2, Qt.KeyboardModifier.KeypadModifier)
    assert document.active_group_id == second.id
    assert document.measurements[0].fiber_group_id == first.id
    QTest.keyClick(canvas, Qt.Key.Key_1)
    assert document.active_group_id == first.id


def test_fullscreen_undo_redo_and_snap_shortcuts_use_real_key_events(counting_window):
    window, document, _first, _second = counting_window
    canvas = window.current_canvas()
    assert window._file_toolbar.isHidden()
    assert window.undo_action.isEnabled()
    QTest.keyClick(canvas, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert len(document.measurements) == 0
    QTest.keyClick(
        canvas,
        Qt.Key.Key_Z,
        Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
    )
    assert len(document.measurements) == 1
    window.set_tool_mode("manual")
    before = window.toggle_edge_snap_action.isChecked()
    QTest.keyClick(canvas, Qt.Key.Key_B)
    assert window.toggle_edge_snap_action.isChecked() != before


def test_fullscreen_shortcuts_preserve_text_editor_undo_and_digits(
    counting_window, desktop_application
):
    window, document, first, _second = counting_window
    editor = QLineEdit(window)
    editor.show()
    editor.setFocus()
    desktop_application.processEvents()
    try:
        QTest.keyClicks(editor, "12")
        assert editor.text() == "12"
        assert document.active_group_id == first.id
        QTest.keyClick(editor, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
        assert editor.text() == ""
        assert len(document.measurements) == 1
    finally:
        editor.deleteLater()
