from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QComboBox, QAbstractSpinBox
from test_live_experiment_content_workspace import workspace


def test_default_count_does_not_select_cancel_or_reclassify(workspace):
    s,c = workspace.session,workspace.controller
    previous=s.add_measurement(20,'wool',field_id=c.field_id)
    c.current_records.append(previous.id)
    workspace._rebuild_records()
    workspace.classification_mode.setCurrentIndex(1)
    c.start((40,40))
    workspace.canvas.setFocus()
    QTest.keyClick(workspace.canvas,Qt.Key.Key_2)
    assert workspace.material.currentData() == 'wool'
    assert s.measurements[previous.id].material_id == 'wool' and c.draft is not None
    assert next(iter(s.count_adjustments.values())).material_id == 'cashmere'
    QTest.keyClick(workspace.canvas,Qt.Key.Key_Control)
    assert workspace.keyboard_mode.currentIndex() == 0
    QTest.keyClick(workspace.canvas,Qt.Key.Key_2)
    assert workspace.material.currentData() == 'cashmere' and c.draft is None
    assert len(s.count_adjustments)==1


def test_ctrl_chords_and_auto_repeat_do_not_toggle_nk_or_repeat_counts(workspace):
    workspace.canvas.setFocus()
    QTest.keyClick(workspace.canvas,Qt.Key.Key_C,Qt.KeyboardModifier.ControlModifier)
    assert workspace.keyboard_mode.currentIndex()==1
    QTest.keyPress(workspace.canvas,Qt.Key.Key_Control)
    repeat=QKeyEvent(QEvent.Type.KeyPress,Qt.Key.Key_Control,Qt.KeyboardModifier.ControlModifier,'',True,1)
    QApplication.sendEvent(workspace.canvas,repeat)
    QTest.keyRelease(workspace.canvas,Qt.Key.Key_Control)
    assert workspace.keyboard_mode.currentIndex()==0
    workspace.keyboard_mode.setCurrentIndex(1)
    repeat=QKeyEvent(QEvent.Type.KeyPress,Qt.Key.Key_2,Qt.KeyboardModifier.NoModifier,'2',True,1)
    QApplication.sendEvent(workspace.canvas,repeat)
    assert not workspace.session.count_adjustments


def wheel(widget, delta=-120):
    event=QWheelEvent(QPointF(5,5),QPointF(widget.mapToGlobal(QPoint(5,5))),QPoint(),QPoint(0,delta),
        Qt.MouseButton.NoButton,Qt.KeyboardModifier.NoModifier,Qt.ScrollPhase.ScrollUpdate,False)
    QApplication.sendEvent(widget,event)


def test_settings_wheel_never_changes_value_even_when_focused(workspace):
    workspace.advanced_toggle.setChecked(True)
    workspace.settings_dialog.activateWindow(); QApplication.processEvents()
    for widget in workspace.settings_dialog.findChildren(QComboBox):
        previous=widget.currentIndex(); widget.setFocus(); wheel(widget)
        assert widget.currentIndex()==previous
    for widget in workspace.settings_dialog.findChildren(QAbstractSpinBox):
        previous=widget.value(); widget.setFocus(); wheel(widget)
        assert widget.value()==previous
    workspace.sample.setFocus(); QTest.keyClicks(workspace.sample,'123')
    QTest.keyClick(workspace.sample,Qt.Key.Key_Control)
    assert workspace.sample.text()=='123' and workspace.keyboard_mode.currentIndex()==1
    assert not workspace.session.count_adjustments


def test_popup_browsing_commits_only_by_explicit_selection(workspace):
    combo=workspace.direction
    previous=combo.currentIndex()
    combo.showPopup(); QApplication.processEvents()
    wheel(combo.view().viewport())
    assert combo.currentIndex()==previous
    combo.hidePopup()


def test_builtin_numeric_dialog_and_trackpad_cannot_mutate_value(workspace):
    from PySide6.QtWidgets import QInputDialog
    dialog=QInputDialog(workspace)
    dialog.setInputMode(QInputDialog.InputMode.DoubleInput)
    dialog.setDoubleValue(1.25); dialog.show()
    editor=dialog.findChild(QAbstractSpinBox)
    editor.setFocus()
    wheel(editor)
    event=QWheelEvent(QPointF(5,5),QPointF(editor.mapToGlobal(QPoint(5,5))),QPoint(0,-19),QPoint(),
        Qt.MouseButton.NoButton,Qt.KeyboardModifier.NoModifier,Qt.ScrollPhase.ScrollUpdate,False)
    QApplication.sendEvent(editor,event)
    assert dialog.doubleValue()==1.25
    dialog.close()
