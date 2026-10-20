"""Real navigation must never route trial history into a hidden image tab."""
from pathlib import Path
import sys

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QComboBox, QLineEdit, QSpinBox

from fdm.geometry import Point
from fdm.live_experiment import default_materials
from fdm.models import ImageDocument

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/validation"))
from validate_workflow_redesign import WorkflowHarness, settle


def add_image_count(window):
    image = QImage(320, 240, QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.white)
    document = ImageDocument(id="shortcut-image", path="shortcut-image.png", image_size=(320, 240))
    document.initialize_runtime_state()
    window._mount_document(document, image, tooltip=document.path)
    document.ensure_default_group()
    window._update_ui_for_current_document()
    window.set_tool_mode("count")
    window._on_canvas_line_committed(document.id, "count", {
        "measurement_kind": "count", "point_px": Point(80, 80)})
    assert len(document.measurements) == 1
    return document


def enter_trial(harness, *, stop=True):
    harness.enter()
    page = harness.select(1)
    page.session.set_materials(default_materials())
    page._refresh_materials()
    page._count_material("wool", 1)
    if stop:
        QTest.mouseClick(harness.hub.capture_button, Qt.MouseButton.LeftButton)
    settle()
    harness.window.activateWindow()
    page.canvas.setFocus()
    settle()
    return page


def history_key(widget, action):
    modifiers = Qt.KeyboardModifier.ControlModifier
    key = Qt.Key.Key_Z
    if action == "redo_y":
        key = Qt.Key.Key_Y
    elif action == "redo_shift_z":
        modifiers |= Qt.KeyboardModifier.ShiftModifier
    QTest.keyClick(widget, key, modifiers)


@pytest.mark.parametrize("stop", [False, True])
@pytest.mark.parametrize("redo", ["redo_y", "redo_shift_z"])
def test_canvas_shortcuts_route_trial_history_while_live_or_stopped(tmp_path, stop, redo):
    with WorkflowHarness(tmp_path, size=(1600, 900)) as harness:
        window = harness.window
        document = add_image_count(window)
        page = enter_trial(harness, stop=stop)
        history_key(page.canvas, "undo")
        assert len(page.session.count_adjustments) == 0
        assert len(document.measurements) == 1
        history_key(page.canvas, redo)
        assert len(page.session.count_adjustments) == 1
        assert len(document.measurements) == 1
        window.measure_workspace_action.trigger()
        window.current_canvas().setFocus()
        settle()
        history_key(window.current_canvas(), "undo")
        assert len(document.measurements) == 0
        assert len(page.session.count_adjustments) == 1
        assert not harness.slot_errors


@pytest.mark.parametrize("redo", ["redo_y", "redo_shift_z"])
def test_trial_redo_does_not_restore_hidden_image_history(tmp_path, redo):
    with WorkflowHarness(tmp_path) as harness:
        document = add_image_count(harness.window)
        harness.window.undo_current_document()
        assert not document.measurements
        page = enter_trial(harness)
        history_key(page.canvas, "undo")
        history_key(page.canvas, redo)
        assert len(page.session.count_adjustments) == 1
        assert not document.measurements
        assert not harness.slot_errors


@pytest.mark.parametrize("action", ["undo", "redo_y", "redo_shift_z"])
def test_candidate_verification_blocks_history_without_touching_hidden_image(tmp_path, monkeypatch, action):
    with WorkflowHarness(tmp_path) as harness:
        document = add_image_count(harness.window)
        page = enter_trial(harness)
        if action != "undo":
            history_key(page.canvas, "undo")
        before = dict(page.session.count_adjustments)
        # Isolate the in-flight candidate guard; no synthetic result is accepted.
        monkeypatch.setattr(type(page.controller), "candidate_pending_count", property(lambda self: 1))
        history_key(page.canvas, action)
        assert page.session.count_adjustments == before
        assert len(document.measurements) == 1
        assert "核验" in page.statusBar().currentMessage()
        monkeypatch.undo()
        assert not harness.slot_errors


def test_floating_records_window_uses_visible_trial_history(tmp_path):
    with WorkflowHarness(tmp_path) as harness:
        document = add_image_count(harness.window)
        page = enter_trial(harness)
        QTest.mouseClick(page.records_toggle, Qt.MouseButton.LeftButton)
        presenter = harness.hub.records
        action = next(a for a in presenter.panel.position_button.menu().actions() if a.text() == "浮动窗口")
        action.trigger()
        presenter.dock.activateWindow()
        presenter.panel.table.setFocus()
        settle()
        assert presenter.dock.isFloating()
        assert QApplication.focusWidget() is presenter.panel.table
        history_key(presenter.panel.table, "undo")
        assert not page.session.count_adjustments
        history_key(presenter.panel.table, "redo_shift_z")
        assert len(page.session.count_adjustments) == 1
        assert len(document.measurements) == 1
        assert not harness.slot_errors


@pytest.mark.parametrize("redo", ["redo_y", "redo_shift_z"])
def test_editor_undo_redo_remains_local_to_text(tmp_path, redo):
    with WorkflowHarness(tmp_path) as harness:
        document = add_image_count(harness.window)
        page = enter_trial(harness)
        editor = QLineEdit(page)
        editor.show()
        editor.setFocus()
        settle()
        QTest.keyClicks(editor, "sample")
        history_key(editor, "undo")
        assert editor.text() == ""
        history_key(editor, redo)
        assert editor.text() == "sample"
        assert len(page.session.count_adjustments) == 1
        assert len(document.measurements) == 1
        editor.deleteLater()
        assert not harness.slot_errors


def test_realtime_stopped_cannot_directly_undo_hidden_image(tmp_path):
    with WorkflowHarness(tmp_path) as harness:
        document = add_image_count(harness.window)
        harness.enter()
        QTest.mouseClick(harness.hub.capture_button, Qt.MouseButton.LeftButton)
        settle()
        assert not harness.window.undo_action.isEnabled()
        harness.window.undo_current_document()
        assert len(document.measurements) == 1


@pytest.mark.parametrize("kind", ["spin", "combo"])
def test_history_toolbar_preserves_native_editing_of_numeric_and_combo_fields(tmp_path, kind):
    with WorkflowHarness(tmp_path) as harness:
        document = add_image_count(harness.window)
        page = enter_trial(harness)
        control = QSpinBox(page) if kind == "spin" else QComboBox(page)
        if kind == "spin":
            control.setRange(0, 99999)
        else:
            control.setEditable(True)
        control.show()
        control.setFocus()
        settle()
        editor = control.lineEdit()
        original = editor.text()
        entered = "123" if kind == "spin" else "sample"
        QTest.keyClick(editor, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
        QTest.keyClicks(editor, entered)
        harness.window.undo_action.trigger()
        assert editor.text() == original
        harness.window.redo_action.trigger()
        assert editor.text() == entered
        assert len(page.session.count_adjustments) == 1
        assert len(document.measurements) == 1
        control.deleteLater()
