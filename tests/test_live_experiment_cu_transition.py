"""CU's frequent category operations stay available without stopping preview."""
import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from fdm.live_experiment.presentation import CU_CATEGORY_COLORS
from test_live_experiment_content_workspace import workspace


def test_right_click_cancels_draft_and_selects_without_extra_count(workspace):
    controller = workspace.controller
    controller.start((30, 30))
    assert controller.draft is not None
    workspace.keyboard_mode.setCurrentIndex(1)  # Count keys do not change menu semantics.
    QTest.mouseClick(workspace.canvas, Qt.MouseButton.RightButton, pos=QPoint(40, 40))
    QApplication.processEvents()
    assert controller.draft is None
    menu = workspace._category_menu
    assert menu.isVisible() and not menu.isModal()
    action = next(action for action in menu.actions() if action.data() == "cashmere")
    before = len(workspace.session.count_adjustments)
    action.trigger()
    menu.close()
    assert workspace.material.currentData() == "cashmere"
    assert workspace.canvas.active_material_id == "cashmere"
    assert len(workspace.session.count_adjustments) == before


def test_right_click_after_selection_changes_only_current_record(workspace):
    session, controller = workspace.session, workspace.controller
    historical = session.add_measurement(22, "wool", field_id="previous-field")
    current = session.add_measurement(23, "wool", field_id=controller.field_id)
    controller.current_records.append(current.id)
    workspace._rebuild_records()
    workspace.classification_mode.setCurrentIndex(1)
    QTest.mouseClick(workspace.canvas, Qt.MouseButton.RightButton, pos=QPoint(40, 40))
    menu = workspace._category_menu
    next(action for action in menu.actions() if action.data() == "cashmere").trigger()
    menu.close()
    assert session.measurements[current.id].material_id == "cashmere"
    assert session.measurements[historical.id].material_id == "wool"
    controller.end_field()
    QTest.mouseClick(workspace.canvas, Qt.MouseButton.RightButton, pos=QPoint(40, 40))
    menu = workspace._category_menu
    next(action for action in menu.actions() if action.data() == "wool").trigger()
    menu.close()
    assert session.measurements[current.id].material_id == "cashmere"


def test_canvas_uses_cu_category_colors_and_keeps_presented_frame(workspace):
    session, controller = workspace.session, workspace.controller
    for x, material in ((60, "wool"), (120, "cashmere")):
        record = session.add_measurement(20, material, field_id=controller.field_id,
                                         endpoints=((x, 40), (x, 180)))
        controller.current_records.append(record.id)
    original = controller.presented.frame
    workspace.canvas.update()
    QApplication.processEvents()
    image = workspace.canvas.grab().toImage().convertToFormat(
        workspace.canvas.image.Format.Format_RGBA8888)
    rgba = np.frombuffer(image.constBits(), np.uint8).reshape(image.height(), image.bytesPerLine())
    pixels = rgba[:, :image.width() * 4].reshape(image.height(), image.width(), 4)
    for color in CU_CATEGORY_COLORS[:2]:
        expected = np.array(QColor(color).getRgb()[:3], np.uint8)
        assert np.count_nonzero(np.all(pixels[:, :, :3] == expected, axis=2)) > 30
    assert controller.presented.frame is original


def test_count_key_mode_is_visible_and_minus_never_removes_measurements(workspace):
    record = workspace.session.add_measurement(20, "wool", field_id=workspace.controller.field_id)
    workspace.controller.current_records.append(record.id)
    assert not workspace.advanced_toggle.isChecked()
    assert workspace.keyboard_mode.isVisibleTo(workspace)
    workspace.keyboard_mode.setCurrentIndex(1)
    workspace.canvas.setFocus()
    QTest.keyClick(workspace.canvas, Qt.Key.Key_1)
    workspace._refresh_summary()
    assert workspace.material_rows["wool"]["extra_count"].text() == "1"
    QTest.keyClick(workspace.canvas, Qt.Key.Key_Minus, Qt.KeyboardModifier.KeypadModifier)
    workspace._refresh_summary()
    assert workspace.material_rows["wool"]["extra_count"].text() == "0"
    QTest.keyClick(workspace.canvas, Qt.Key.Key_Minus)
    assert workspace.session.measurements[record.id].diameter_um == 20
    workspace.sample.setFocus()
    workspace.sample.clear()
    QTest.keyClicks(workspace.sample, "1-9")
    assert workspace.sample.text() == "1-9"
