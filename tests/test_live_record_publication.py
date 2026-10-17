"""Publishing rows must not lay out the collapsed records panel per fiber."""
from unittest.mock import Mock

import pytest
from PySide6.QtWidgets import QApplication

from test_live_experiment_content_workspace import workspace


def add_records(view, count, *, batch=False):
    records = [view.session.add_measurement(20 + index / 100, "wool",
        field_id=view.controller.field_id) for index in range(count)]
    view.controller.current_records.extend(record.id for record in records)
    if batch:
        view._records_accepted(records)
    else:
        for record in records:
            view._record_accepted(record)
    return [record.id for record in records]


@pytest.mark.parametrize("batch", [False, True])
def test_hidden_publication_keeps_models_current_and_follows_latest_on_show(workspace, batch):
    table = workspace.current_table
    table.scrollToBottom = Mock(wraps=table.scrollToBottom)
    assert not table.isVisible()
    ids = add_records(workspace, 60, batch=batch)
    # Domain and both models publish synchronously; only invisible scrolling waits.
    assert workspace.current_model.ids == workspace.history_model.ids == ids
    table.scrollToBottom.assert_not_called()
    workspace.records_toggle.setChecked(True)
    QApplication.processEvents()
    QApplication.processEvents()
    assert table.isVisible()
    table.scrollToBottom.assert_called_once()
    assert table.verticalScrollBar().maximum() > 0
    assert table.verticalScrollBar().value() == table.verticalScrollBar().maximum()
    add_records(workspace, 1)
    assert table.scrollToBottom.call_count == 2
    assert table.verticalScrollBar().value() == table.verticalScrollBar().maximum()


def test_off_tab_records_follow_once_on_return_and_new_field_has_no_stale_rows(workspace):
    workspace.records_toggle.setChecked(True)
    workspace.tabs.setCurrentIndex(1)
    QApplication.processEvents()
    table = workspace.current_table
    table.scrollToBottom = Mock(wraps=table.scrollToBottom)
    ids = add_records(workspace, 60)
    table.scrollToBottom.assert_not_called()
    workspace.tabs.setCurrentIndex(0)
    QApplication.processEvents()
    QApplication.processEvents()
    table.scrollToBottom.assert_called_once()
    assert table.verticalScrollBar().value() == table.verticalScrollBar().maximum()
    workspace.records_toggle.setChecked(False)
    more = add_records(workspace, 1)
    workspace.controller.end_field("test movement")
    workspace.records_toggle.setChecked(True)
    QApplication.processEvents()
    QApplication.processEvents()
    assert workspace.current_model.rowCount() == 0
    assert workspace.history_model.ids == ids + more
    assert table.verticalScrollBar().maximum() == 0
