import hashlib

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QColorDialog

from fdm.geometry import Line, Point
from fdm.models import ImageDocument, Measurement, ObjectAppearanceOverride
from fdm.ui.measurement_results_model import MeasurementResultColumn
from test_metrology_workspace import mounted_window, professional_slide


def add_records(window, document, *, same_group=False):
    one = document.create_group(color="#446688", label="样品甲")
    two = document.create_group(color="#884466", label="样品乙")
    area = Measurement("area-a", document.id, one.id, "manual", "area",
                       polygon_px=[Point(10, 10), Point(25, 10), Point(10, 25)],
                       appearance=ObjectAppearanceOverride(stroke_width=4))
    line = Measurement("line-b", document.id, one.id if same_group else two.id, "manual",
                       line_px=Line(Point(15, 10), Point(50, 50)),
                       appearance=ObjectAppearanceOverride(text_color="#224466"))
    count = Measurement("count-c", document.id, two.id, "manual", "count", point_px=Point(20, 25),
                        appearance=ObjectAppearanceOverride(marker_scale=2))
    for record in (area, line, count):
        document.add_measurement(record)
    window._update_ui_for_current_document()
    return one, two


def select_visible(window, ids, *, shift=False):
    records = window._records_controller
    table = window._bottom_records_pane.table
    window._adaptive_layout.set_panel_visible("results", True)
    window._results_tabs.setCurrentWidget(window._bottom_records_pane)
    window._results_dock.raise_()
    from PySide6.QtWidgets import QApplication
    QApplication.processEvents()
    for i, identity in enumerate(ids):
        row = records.source_model.source_row_for_id(identity)
        proxy = records.proxy_model.mapFromSource(records.source_model.index(row, int(MeasurementResultColumn.RESULT)))
        table.scrollTo(proxy)
        modifier = Qt.KeyboardModifier.NoModifier if not i else (
            Qt.KeyboardModifier.ShiftModifier if shift else Qt.KeyboardModifier.ControlModifier)
        QTest.mouseClick(table.viewport(), Qt.MouseButton.LeftButton, modifier, table.visualRect(proxy).center())


def test_sorted_ctrl_selection_applies_only_explicit_fields_in_one_undo(mounted_window, monkeypatch):
    window, doc, _, _ = mounted_window()
    one, two = add_records(window, doc)
    controller = window._records_controller
    controller.set_sort(MeasurementResultColumn.RESULT, Qt.SortOrder.DescendingOrder)
    select_visible(window, ["area-a", "line-b"])
    selected = {"area-a", "line-b"}
    assert set(controller.selected_measurement_ids()) == selected
    editor = window._object_inspector.batch_editor
    assert editor.isVisibleTo(window._object_inspector)
    assert editor._group.currentIndex() == -1
    assert editor._buttons["stroke_width"].text() == "多种值"
    assert editor._buttons["text_color"].text() == "多种值"
    assert editor._buttons["marker_scale"].isHidden()
    assert editor._buttons["font_family"].text().startswith("继承（")
    assert editor._buttons["stroke_color"].text() == "继承（多种值）"
    assert window._current_measurement_summary.groupCombo.currentIndex() == -1
    before = {m.id: m.to_dict() for m in doc.measurements}
    geometry_revision = doc.measurement_geometry_revision
    history_count = doc.history.command_count
    monkeypatch.setattr(QColorDialog, "getColor", lambda *args: QColor("#bb2288"))
    editor._buttons["stroke_color"].click()
    index = editor._group.findData(two.id)
    editor._group.setCurrentIndex(index)
    editor._group.activated.emit(index)
    window._refresh_object_inspector()
    assert editor._apply.isEnabled()
    assert {m.id: m.to_dict() for m in doc.measurements} == before
    editor._apply.click()
    assert set(controller.selected_measurement_ids()) == selected
    assert doc.history.command_count == history_count + 1
    assert doc.measurement_geometry_revision == geometry_revision
    assert set(two.measurement_ids) == {"area-a", "line-b", "count-c"}
    assert doc.get_measurement("area-a").appearance.stroke_width == 4
    assert doc.get_measurement("line-b").appearance.text_color == "#224466"
    assert doc.get_measurement("line-b").appearance.stroke_color.upper() == "#BB2288"
    assert doc.get_measurement("count-c").to_dict() == before["count-c"]
    reopened = ImageDocument.from_dict(doc.to_dict())
    assert reopened.get_measurement("area-a").appearance == doc.get_measurement("area-a").appearance
    assert reopened.get_measurement("area-a").fiber_group_id == two.id
    window.undo_current_document()
    assert set(controller.selected_measurement_ids()) == selected
    assert {m.id: m.to_dict() for m in doc.measurements} == before
    window.redo_current_document()
    assert editor._group.currentData() == two.id
    editor._inherit.click()
    editor._apply.click()
    assert all(doc.get_measurement(i).appearance is None for i in selected)
    assert all(doc.get_measurement(i).fiber_group_id == two.id for i in selected)
    assert doc.get_measurement("count-c").appearance.marker_scale == 2


def test_category_filter_removes_rows_without_retargeting_old_selection(mounted_window):
    window, doc, _, _ = mounted_window()
    one, two = add_records(window, doc, same_group=True)
    controller = window._records_controller
    controller.set_filters(group=one.label)
    select_visible(window, ["area-a", "line-b"], shift=True)
    assert set(controller.selected_measurement_ids()) == {"area-a", "line-b"}
    summary = window._current_measurement_summary
    old_target = summary.batch_target
    assert summary.groupCombo.currentData() == one.id
    index = summary.groupCombo.findData(None)
    summary.groupCombo.setCurrentIndex(index)
    summary.groupCombo.activated.emit(index)
    assert controller.filters[2] == one.label and controller.proxy_model.rowCount() == 0
    assert controller.selected_measurement_ids() == []
    assert doc.view_state.selected_measurement_id is None
    count = doc.history.command_count
    window._edit_selected_measurements(*old_target, {"fiber_group_id": two.id})
    assert doc.history.command_count == count
    assert not one.measurement_ids and two.measurement_ids == ["count-c"]
    assert not summary.groupCombo.isEnabled()
    window.undo_current_document()
    assert controller.proxy_model.rowCount() == 2


def test_slide_mixed_record_types_keep_pixels_and_hidden_fields(professional_slide):
    window, doc, _, path, initial_sha = professional_slide
    add_records(window, doc)
    controller = window._records_controller
    controller.select_measurement_ids(("line-b", "count-c"), primary="line-b")
    editor = window._object_inspector.batch_editor
    assert editor._buttons["marker_scale"].isHidden() and editor._buttons["stroke_width"].isHidden()
    assert editor._buttons["font_size"].text().startswith("继承（")
    doc_stamp = doc.state_stamp
    window._edit_selected_measurements(doc.id, ("line-b", "count-c"), {"stroke_width": 3})
    assert doc.state_stamp == doc_stamp
    editor._set_field("font_size", 27)
    editor._apply.click()
    assert doc.get_measurement("line-b").appearance.font_size == 27
    assert doc.get_measurement("count-c").appearance.marker_scale == 2
    window.undo_current_document()
    assert doc.get_measurement("line-b").appearance.font_size is None
    assert hashlib.sha256(path.read_bytes()).hexdigest() == initial_sha
