from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QStyleOptionViewItem

from fdm.geometry import Line, Point
from fdm.models import Measurement, ProjectCompatibilityState
from fdm.ui.measurement_results_model import MeasurementResultColumn as Col
from test_workbench_workflows import mount, settle, window


def add_line(window, doc, identity, length):
    item = Measurement(identity, doc.id, doc.active_group_id, "manual",
                       line_px=Line(Point(0, 5), Point(length, 5)), status="manual")
    window._append_new_measurement(doc, item, label="新增测量")
    return item


def setup_table(window):
    window.resize(1366, 768)
    doc = mount(window, 1)
    window._adaptive_layout.set_panel_visible("inspector", True)
    window._inspector_dock.raise_()
    settle()
    return doc, window._inspector_records_pane


def test_compact_delegates_combine_native_values_and_flag_review(window):
    doc, pane = setup_table(window)
    doc.measurements[0].status = "manual_review"
    first = pane.table.model().index(0, Col.RESULT_SEQUENCE)
    option = QStyleOptionViewItem()
    pane.table.itemDelegateForIndex(first).initStyleOption(option, first)
    assert option.text == "线段 1"
    assert not option.icon.isNull()
    assert "需人工复核" in first.data(Qt.ItemDataRole.ToolTipRole)
    result = first.siblingAtColumn(Col.RESULT)
    pane.table.itemDelegateForIndex(result).initStyleOption(option, result)
    assert option.text == f"{result.data()} μm"
    assert first.siblingAtColumn(Col.UNIT).data() == "um"
    assert option.displayAlignment & Qt.AlignmentFlag.AlignRight
    area = Measurement("area", doc.id, None, "polygon_area", "area",
                       polygon_px=[Point(1, 1), Point(20, 1), Point(20, 20)])
    window._apply_document_change(doc, "添加面积", lambda: doc.add_measurement(area))
    area_cell = pane.controller.selection_model.currentIndex().siblingAtColumn(Col.RESULT)
    pane.table.itemDelegateForIndex(area_cell).initStyleOption(option, area_cell)
    assert pane.controller.selected_measurement_ids() == [area.id]
    assert option.text == f"{area_cell.data()} μm²"
    assert area_cell.siblingAtColumn(Col.UNIT).data() == "um²"


def test_append_respects_popup_sort_and_hidden_new_records(window):
    doc, pane = setup_table(window)
    controller = pane.controller
    original = doc.measurements[0]
    group = doc.fiber_groups[1]
    new_measurement_category = doc.active_group_id
    controller.select_measurement_id(original.id)
    group_cell = controller.proxy_model.index(0, Col.GROUP)
    QTest.mouseClick(pane.table.viewport(), Qt.MouseButton.LeftButton, pos=pane.table.visualRect(group_cell).center())
    editor = pane.table.indexWidget(group_cell)
    assert editor is not None and editor.view().isVisible()
    new = add_line(window, doc, "arrived-during-edit", 20)
    assert controller.selected_measurement_ids() == [original.id]
    assert pane.table.indexWidget(group_cell) is editor
    choice = editor.findData(group.id)
    option = editor.model().index(choice, 0)
    QTest.mouseClick(editor.view().viewport(), Qt.MouseButton.LeftButton,
                     pos=editor.view().visualRect(option).center())
    settle()
    assert original.fiber_group_id == group.id
    assert new.fiber_group_id == new_measurement_category
    assert doc.active_group_id == new_measurement_category
    assert controller.selected_measurement_ids() == [new.id]

    controller.set_sort(Col.RESULT, Qt.SortOrder.AscendingOrder)
    smaller = add_line(window, doc, "sorts-first", 1)
    assert controller.selection_model.currentIndex().row() == 0
    assert controller.selected_measurement_ids() == [smaller.id]
    controller.set_filters(group=group.label)
    controller.select_measurement_id(original.id)
    filtered = add_line(window, doc, "filtered", 50)
    assert controller.proxy_model.rowCount() == 1
    assert controller.selected_measurement_ids() == []
    assert doc.view_state.selected_measurement_id == filtered.id
    assert pane.filter_button.text() == "筛选 (1)"
    assert "新增 1 条，当前筛选未显示" in pane.count_label.text()
    pane.filter_button.menu().actions()[-1].trigger()
    assert controller.filters == ("", "", "", "")


def test_batch_category_menu_uses_frozen_ids_and_one_undo(window):
    doc, pane = setup_table(window)
    group_a, group_b = doc.fiber_groups
    doc.set_active_group(group_a.id)
    a = add_line(window, doc, "older", 20)
    b = add_line(window, doc, "newer", 30)
    pane.controller.set_filters(group=group_a.label)
    for row, modifier in [(1, Qt.KeyboardModifier.NoModifier), (2, Qt.KeyboardModifier.ControlModifier)]:
        column = Col.RESULT if row == 1 else Col.GROUP
        cell = pane.controller.proxy_model.index(row, column)
        QTest.mouseClick(pane.table.viewport(), Qt.MouseButton.LeftButton, modifier,
                         pos=pane.table.visualRect(cell).center())
    assert pane.table.indexWidget(cell) is None
    assert pane.batch_group_button.isVisible()
    previous_commands = doc.history.command_count
    menu = pane.batch_group_button.menu()
    menu.popup(pane.batch_group_button.mapToGlobal(pane.batch_group_button.rect().bottomLeft()))
    settle()
    action = next(action for action in menu.actions() if group_b.label in action.text())
    QTest.mouseClick(menu, Qt.MouseButton.LeftButton, pos=menu.actionGeometry(action).center())
    settle()
    assert doc.history.command_count == previous_commands + 1
    assert a.fiber_group_id == b.fiber_group_id == group_b.id
    assert doc.measurements[0].fiber_group_id == group_a.id
    assert doc.active_group_id == group_a.id
    assert pane.controller.filters[2] == group_a.label
    assert pane.controller.proxy_model.rowCount() == 1
    window.undo_current_document()
    assert doc.get_measurement(a.id).fiber_group_id == doc.get_measurement(b.id).fiber_group_id == group_a.id


def test_cancel_open_category_popup_reveals_pending_record(window):
    doc, pane = setup_table(window)
    controller = pane.controller
    original = doc.measurements[0]
    original_group = original.fiber_group_id
    group_cell = controller.proxy_model.index(0, Col.GROUP)
    QTest.mouseClick(pane.table.viewport(), Qt.MouseButton.LeftButton, pos=pane.table.visualRect(group_cell).center())
    editor = pane.table.indexWidget(group_cell)
    new = add_line(window, doc, "arrived-before-escape", 24)
    QTest.keyClick(editor.view(), Qt.Key.Key_Down)
    QTest.keyClick(editor.view(), Qt.Key.Key_Escape)
    settle()
    assert original.fiber_group_id == original_group
    assert pane.table.indexWidget(group_cell) is None
    assert controller.selected_measurement_ids() == [new.id]


def test_dismissing_popup_without_selecting_releases_pending_record(window):
    doc, pane = setup_table(window)
    controller = pane.controller
    original = doc.measurements[0]
    group_before = original.fiber_group_id
    cell = controller.proxy_model.index(0, Col.GROUP)
    QTest.mouseClick(pane.table.viewport(), Qt.MouseButton.LeftButton, pos=pane.table.visualRect(cell).center())
    editor = pane.table.indexWidget(cell)
    new = add_line(window, doc, "arrived-before-click-away", 24)
    popup = editor.view().window()
    outside = popup.mapFromGlobal(pane.search_edit.mapToGlobal(pane.search_edit.rect().center()))
    assert not popup.rect().contains(outside)
    QTest.mouseClick(popup.windowHandle(), Qt.MouseButton.LeftButton, pos=outside)
    settle()
    assert original.fiber_group_id == group_before
    assert pane.table.indexWidget(cell) is None
    assert controller.selected_measurement_ids() == [new.id]


def test_delete_menu_and_buttons_follow_selection_and_compatibility(window):
    doc, pane = setup_table(window)
    bottom = window._bottom_records_pane
    original = doc.measurements[0]
    pane.controller.select_measurement_id(None)
    for view in (pane, bottom):
        assert not view.delete_selected_button.isEnabled()
        assert view.delete_all_button.isEnabled()
        assert view.delete_category_button.isEnabled()
    pane.controller.select_measurement_id(original.id)
    assert pane.delete_selected_button.isEnabled() and bottom.delete_selected_button.isEnabled()
    pane.delete_selected_button.trigger()
    assert not doc.measurements
    for view in (pane, bottom):
        assert not any(button.isEnabled() for button in (
            view.delete_selected_button, view.delete_all_button, view.delete_category_button
        ))
    window.undo_current_document()
    pane.controller.select_measurement_id(original.id)
    window.project.compatibility = ProjectCompatibilityState(unknown_required_features=("new/v1",))
    window._update_action_states()
    for view, trigger in ((pane, "trigger"), (bottom, "click")):
        for button in (view.delete_selected_button, view.delete_all_button, view.delete_category_button):
            assert not button.isEnabled()
            getattr(button, trigger)()
    assert [record.id for record in doc.measurements] == [original.id]


def test_category_edits_opened_before_read_only_do_not_write(window):
    doc, pane = setup_table(window)
    original = doc.measurements[0]
    group_a, group_b = doc.fiber_groups
    doc.set_active_group(group_a.id)
    other = add_line(window, doc, "read-only-check", 20)
    controller = pane.controller
    controller.select_measurement_ids((original.id, other.id))
    menu = pane.batch_group_button.menu()
    menu.popup(pane.batch_group_button.mapToGlobal(pane.batch_group_button.rect().bottomLeft()))
    batch_action = next(action for action in menu.actions() if group_b.label in action.text())
    readonly = ProjectCompatibilityState(unknown_required_features=("new/v1",))
    window.project.compatibility = readonly
    window._update_action_states()
    window._refresh_object_inspector()
    assert not pane.batch_group_button.isVisible()
    batch_action.trigger()
    menu.hide()
    settle()
    assert [record.fiber_group_id for record in doc.measurements] == [group_a.id, group_a.id]

    window.project.compatibility = ProjectCompatibilityState()
    window._update_action_states()
    controller.select_measurement_id(original.id)
    group_cell = controller.proxy_model.index(0, Col.GROUP)
    QTest.mouseClick(pane.table.viewport(), Qt.MouseButton.LeftButton, pos=pane.table.visualRect(group_cell).center())
    editor = pane.table.indexWidget(group_cell)
    assert editor is not None
    window.project.compatibility = readonly
    window._update_action_states()
    assert not (group_cell.flags() & Qt.ItemFlag.ItemIsEditable)
    option = editor.model().index(editor.findData(group_b.id), 0)
    QTest.mouseClick(editor.view().viewport(), Qt.MouseButton.LeftButton,
                     pos=editor.view().visualRect(option).center())
    settle()
    assert doc.get_measurement(original.id).fiber_group_id == group_a.id
    assert not controller.proxy_model.setData(group_cell, group_b.id)


def test_return_to_latest_after_reading_older_row_or_adding_to_inactive_image(window):
    doc, pane = setup_table(window)
    for i in range(2, 41):
        doc.add_measurement(Measurement(f"record-{i}", doc.id, None, "manual",
                                        line_px=Line(Point(0, i), Point(40, i))))
    window._update_ui_for_current_document()
    pane.controller.select_measurement_id("record-5")
    pane.scroll_to_measurement("record-5")
    newest = add_line(window, doc, "record-41", 20)
    settle()
    assert pane.controller.selected_measurement_ids() == [newest.id]
    last = pane.controller.proxy_model.index(40, Col.RESULT)
    assert pane.table.viewport().rect().contains(pane.table.visualRect(last).center())

    other = mount(window, 2)
    window._set_current_document(doc.id)
    pane.controller.select_measurement_id("record-5")
    pane.scroll_to_measurement("record-35")
    before = (pane.table.verticalScrollBar().value(), pane.controller.selected_measurement_ids())
    add_line(window, other, "background", 5)
    settle()
    assert before == (pane.table.verticalScrollBar().value(), pane.controller.selected_measurement_ids())
    added = Measurement("background-import", other.id, None, "manual",
                        line_px=Line(Point(0, 4), Point(24, 4)))
    window._apply_document_change(other, "后台导入测量", lambda: other.add_measurement(added))
    settle()
    assert before == (pane.table.verticalScrollBar().value(), pane.controller.selected_measurement_ids())
    window._apply_document_change(other, "清除后台识别结果", lambda: other.remove_measurements([added.id]))
    settle()
    assert before == (pane.table.verticalScrollBar().value(), pane.controller.selected_measurement_ids())
