"""Property edits start from real records or professional result-table selections."""
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMenu

from fdm.geometry import Line, Point
from fdm.measurement_annotation import MeasurementAnnotationLayout
from fdm.ui.rendering import resolve_measurement_annotation
from test_measurement_batch_properties import add_records, select_visible
from test_metrology_workspace import _distance, mounted_window


def choose(editor, key, value):
    combo = editor._fields[key]
    for index in range(combo.count()):
        if combo.itemData(index) == value:
            combo.setCurrentIndex(index)
            return
    raise AssertionError(f"Missing {key} option {value}")


def test_properties_starts_at_basic_controls_and_annotation_shortcut_reveals_group(mounted_window, desktop_application):
    window, doc, _, _ = mounted_window()
    window.set_tool_mode("select")
    add_records(window, doc)
    inspector = window._object_inspector
    scroll = window._inspector_scroll
    scroll.setMaximumHeight(160)
    for identities, first, last in (
        (["line-b"], inspector._summary_label, inspector._reset_button),
        (["line-b", "area-a"], inspector.batch_editor._summary, inspector.batch_editor._apply),
    ):
        select_visible(window, identities)
        window._inspector_records_pane.properties_button.click()
        desktop_application.processEvents()
        assert scroll.verticalScrollBar().value() == 0
        assert scroll.viewport().rect().contains(first.mapTo(scroll.viewport(), QPoint(1, 1)))
        assert last.mapTo(inspector, last.rect().bottomLeft()).y() < inspector.annotation_editor.y()
        window._show_result_annotation_properties()
        desktop_application.processEvents()
        assert scroll.verticalScrollBar().value() > 0
        assert scroll.viewport().rect().contains(inspector.annotation_editor.mapTo(scroll.viewport(), QPoint(1, 3)))
        window._return_to_records_button.click()
        window._inspector_records_pane.properties_button.click()
        desktop_application.processEvents()
        assert scroll.verticalScrollBar().value() == 0
        window._return_to_records_button.click()


def test_selection_leaves_inspector_visibility_and_scroll_to_user(mounted_window, desktop_application, monkeypatch):
    window, doc, _, _ = mounted_window()
    window.set_tool_mode("select")
    add_records(window, doc)
    line = doc.get_measurement("line-b")
    line.annotation_layout = MeasurementAnnotationLayout.ordinary_near()
    editor = window._object_inspector.annotation_editor
    canvas = window.current_canvas()
    window._inspector_dock.raise_()
    desktop_application.processEvents()

    assert window._inspector_pages.currentWidget() is window._records_page
    select_visible(window, ["line-b", "area-a"])
    desktop_application.processEvents()
    assert set(editor.target[2]) == {"line-b", "area-a"}
    assert window._inspector_pages.currentWidget() is window._records_page
    # A click on the canvas label should select that record, without scrolling to its editor.
    select_visible(window, ["area-a"])
    desktop_application.processEvents()
    scroll = window._inspector_scroll.verticalScrollBar()
    scroll.setValue(scroll.maximum())
    previous_scroll = scroll.value()
    label = resolve_measurement_annotation(line, doc, canvas._settings, canvas.image_to_widget)
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=label.text_rect.center().toPoint())
    desktop_application.processEvents()
    assert editor.target[2] == (line.id,)
    assert window._inspector_pages.currentWidget() is window._records_page and scroll.value() == previous_scroll

    class ChooseProperties(QMenu):
        def exec(self, *_):
            return next(action for action in self.actions() if action.text() == "标注属性…")

    monkeypatch.setattr("fdm.ui.canvas.QMenu", ChooseProperties)
    QTest.mouseClick(canvas, Qt.MouseButton.RightButton, pos=label.text_rect.center().toPoint())
    desktop_application.processEvents()
    assert window._inspector_pages.currentWidget() is window._properties_page
    select_visible(window, ["area-a"])
    assert window._inspector_pages.currentWidget() is window._properties_page and editor.target[2] == ("area-a",)
    window._return_to_records_button.click()
    select_visible(window, ["line-b"])
    assert window._inspector_pages.currentWidget() is window._records_page


def test_ordinary_named_default_reaches_actual_creation_and_restore_only_changes_position(mounted_window):
    window, doc, workspace, _ = mounted_window()
    line = Line(Point(20, 25), Point(75, 50))
    window._on_canvas_line_committed(doc.id, "manual", line)
    first = doc.measurements[-1]
    workspace.legacy_properties(doc.id, first.id)
    editor = window._object_inspector.annotation_editor
    assert editor.default_button.text() == "设为普通直径线默认"
    choose(editor, "show_identifier", True)
    choose(editor, "style", "dual")
    editor.default_button.click()
    assert first.annotation_layout.show_identifier
    window._on_canvas_line_committed(doc.id, "manual", Line(Point(80, 50), Point(135, 50)))
    second = doc.measurements[-1]
    assert second.annotation_layout.show_identifier
    assert second.annotation_layout.leader_style == "dual" and second.annotation_layout.show_leader
    assert second.annotation_layout.label_position == "external"
    fields = window._app_settings.annotation_style_defaults["line"]
    assert "offset_px" not in fields
    workspace.legacy_properties(doc.id, first.id)
    editor.restore_button.click()
    assert first.annotation_layout.label_position == "near"
    assert first.annotation_layout.offset_px is None and first.annotation_layout.show_identifier
    assert second.annotation_layout.label_position == "external"


def test_mixed_ordinary_selection_preserves_per_object_manual_offsets(mounted_window):
    window, doc, workspace, _ = mounted_window()
    add_records(window, doc)
    area, line = doc.get_measurement("area-a"), doc.get_measurement("line-b")
    area.annotation_layout = MeasurementAnnotationLayout(show_identifier=True).with_manual_offset(Point(12, 13))
    line.annotation_layout = MeasurementAnnotationLayout().with_manual_offset(Point(89, 50))
    select_visible(window, [area.id, line.id])
    editor = window._object_inspector.annotation_editor
    assert editor._fields["show_identifier"].currentIndex() == -1
    assert not editor.default_button.isEnabled()
    assert editor._fields["style"].findData("dual") == -1
    choose(editor, "show_metric_name", True)
    count = doc.history.command_count
    editor.apply_button.click()
    assert doc.history.command_count == count + 1
    for record, offset, identifier in ((area, Point(12, 13), True), (line, Point(89, 50), False)):
        assert record.annotation_layout.offset_px == offset and record.annotation_layout.show_metric_name
        assert record.annotation_layout.show_identifier is identifier
    window.undo_current_document()
    assert not doc.get_measurement(line.id).annotation_layout.show_metric_name


def test_professional_multiselection_uses_unique_definition_ids_and_one_undo(mounted_window):
    window, doc, workspace, _ = mounted_window()
    one = _distance(workspace, (20, 20), (65, 40))
    two = _distance(workspace, (110, 140), (190, 120))
    table = workspace.panel.table
    rows = {identity: [row for row in range(table.rowCount()) if table.item(row, 0).data(Qt.ItemDataRole.UserRole) == identity]
            for identity in (one.id, two.id)}
    workspace.dock.raise_()
    for row, modifier in ((rows[one.id][0], Qt.KeyboardModifier.NoModifier),
                          (rows[one.id][1], Qt.KeyboardModifier.ControlModifier),
                          (rows[two.id][0], Qt.KeyboardModifier.ControlModifier)):
        item = table.item(row, 0)
        table.scrollToItem(item)
        QTest.mouseClick(table.viewport(), Qt.MouseButton.LeftButton, modifier, table.visualItemRect(item).center())
    editor = window._object_inspector.annotation_editor
    assert set(editor.target[2]) == {one.id, two.id}
    assert len(editor.targets) == 2
    assert editor.default_button.isEnabled()
    assert window._inspector_pages.currentWidget() is window._records_page
    workspace.panel.properties_button.click()
    assert window._inspector_pages.currentWidget() is window._properties_page
    inspector = window._object_inspector
    assert inspector._empty_label.isVisible()
    assert inspector._empty_label.geometry().bottom() < editor.y()
    assert inspector._empty_label.height() <= inspector._empty_label.sizeHint().height() + 4
    choose(editor, "show_identifier", True)
    choose(editor, "metric_key", "dy")
    count = doc.history.command_count
    geo = doc.professional_geometry_revision
    editor.apply_button.click()
    for identity in (one.id, two.id):
        saved = doc.get_professional_measurement(identity).annotation_layout
        assert saved.show_identifier and saved.metric_key == "dy"
    assert doc.history.command_count == count + 1 and doc.professional_geometry_revision == geo
    assert set(editor.target[2]) == {one.id, two.id}
    window.undo_current_document()
    assert not doc.get_professional_measurement(two.id).annotation_layout.show_identifier


def test_arrange_professional_keeps_ordinary_and_manual_labels(mounted_window):
    window, doc, workspace, _ = mounted_window()
    _distance(workspace, (25, 25), (66, 51))
    manual = _distance(workspace, (120, 110), (190, 160))
    layout = manual.annotation_layout.with_manual_offset(Point(22, 26))
    workspace.edit_layout(workspace.controller(), manual.id, layout)
    window._on_canvas_line_committed(doc.id, "manual", Line(Point(10, 36), Point(35, 41)))
    ordinary = doc.measurements[-1]
    ordinary.annotation_layout = ordinary.annotation_layout.with_manual_offset(Point(-33, 44))
    before = ordinary.to_dict()
    workspace.arrange()
    assert doc.get_measurement(ordinary.id).to_dict() == before
    assert doc.get_professional_measurement(manual.id).annotation_layout == layout


def test_professional_metric_controls_follow_actual_witness(mounted_window):
    from test_metrology_workspace import _begin, _click
    window, doc, workspace, _ = mounted_window()
    _begin(workspace, "circle")
    for point in [(110, 70), (70, 110), (30, 70)]:
        _click(workspace.controller().canvas, point)
    circle = doc.professional_measurements[-1]
    workspace.professional_properties(workspace.controller(), circle)
    editor = window._object_inspector.annotation_editor
    choose(editor, "metric_key", "diameter")
    assert editor._fields["style"].findData("dual") >= 0
    assert not editor._fields["show_symbol"].isHidden()
    choose(editor, "metric_key", "center_x")
    assert editor._fields["style"].findData("dual") < 0
    assert editor._fields["show_symbol"].isHidden()
    editor.apply_button.click()
    assert doc.get_professional_measurement(circle.id).annotation_layout.metric_key == "center_x"


def test_filtered_record_cannot_apply_old_annotation_draft(mounted_window):
    window, doc, workspace, _ = mounted_window()
    one, two = add_records(window, doc)
    select_visible(window, ["line-b"])
    editor = window._object_inspector.annotation_editor
    old_selection = editor.target
    choose(editor, "show_identifier", True)
    window._records_controller.set_filters(group=one.label)
    assert "line-b" not in window._records_controller.selected_measurement_ids()
    before = [m.to_dict() for m in doc.measurements]
    workspace.edit_annotation_selection(old_selection, {"show_identifier": True}, "apply")
    assert [m.to_dict() for m in doc.measurements] == before
