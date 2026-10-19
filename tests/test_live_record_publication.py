"""Publishing rows updates the shared records view without hidden scrolling."""
from unittest.mock import Mock

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
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


def settle():
    for _ in range(8):
        QApplication.processEvents()


def toggle_records(view):
    QTest.mouseClick(view.records_toggle, Qt.MouseButton.LeftButton)
    settle()


def current_field_filter(panel):
    combo = panel.field_filter
    QTest.mouseClick(combo, Qt.MouseButton.LeftButton)
    QTest.keyClick(combo.view(), Qt.Key.Key_End)
    QTest.keyClick(combo.view(), Qt.Key.Key_Return)
    settle()
    assert combo.currentData() is True


@pytest.mark.parametrize("batch", [False, True])
def test_hidden_publication_keeps_models_current_and_follows_latest_on_show(workspace, batch):
    # Establish the current trial through its real entry, then hide the dock.
    toggle_records(workspace)
    presenter = workspace._records_presenter()
    assert presenter.is_open
    toggle_records(workspace)
    table = presenter.table
    table.scrollToBottom = Mock(wraps=table.scrollToBottom)
    assert not table.isVisible()
    ids = add_records(workspace, 60, batch=batch)
    # Both compatibility models and the unified ledger publish synchronously;
    # only invisible scrolling waits until the records dock is shown.
    assert workspace.current_model.ids == workspace.history_model.ids == ids
    assert presenter.model.keys == [("measurement", key) for key in ids]
    settle()
    table.scrollToBottom.assert_not_called()
    toggle_records(workspace)
    assert table.isVisible()
    table.scrollToBottom.assert_called_once()
    assert table.verticalScrollBar().maximum() > 0
    assert table.verticalScrollBar().value() == table.verticalScrollBar().maximum()
    add_records(workspace, 1)
    settle()
    assert table.scrollToBottom.call_count == 2
    assert table.verticalScrollBar().value() == table.verticalScrollBar().maximum()


def test_hidden_records_follow_on_return_and_current_field_filter_has_no_stale_rows(workspace):
    toggle_records(workspace)
    presenter = workspace._records_presenter()
    current_field_filter(presenter.panel)
    toggle_records(workspace)
    table = presenter.table
    table.scrollToBottom = Mock(wraps=table.scrollToBottom)
    ids = add_records(workspace, 60)
    settle()
    table.scrollToBottom.assert_not_called()
    toggle_records(workspace)
    table.scrollToBottom.assert_called_once()
    assert table.verticalScrollBar().value() == table.verticalScrollBar().maximum()
    assert presenter.proxy.rowCount() == len(ids)
    current_field_filter(presenter.panel)
    toggle_records(workspace)
    more = add_records(workspace, 1)
    workspace.controller.end_field("test movement")
    settle()
    assert workspace.current_model.rowCount() == 0
    assert workspace.history_model.ids == ids + more
    assert presenter.model.keys == [("measurement", key) for key in ids + more]
    assert presenter.proxy.rowCount() == 0
    toggle_records(workspace)
    # Reopening defaults to all fields; the explicit current-field filter
    # switches to the new empty field without losing historical records.
    assert presenter.proxy.rowCount() == len(ids + more)
    current_field_filter(presenter.panel)
    assert presenter.proxy.rowCount() == 0
    assert table.verticalScrollBar().maximum() == 0
