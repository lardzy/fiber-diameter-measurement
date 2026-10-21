"""Publishing rows updates the shared records view without hidden scrolling."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from test_live_experiment_content_workspace import make_session, workspace


def add_records(view, count, *, batch=False):
    offset = len(view.session.measurements)
    records = [view.session.add_measurement(20 + index / 100, "wool",
        field_id=view.controller.field_id) for index in range(count)]
    # Publication/scrolling tests need chronological rows. Windows may return
    # identical wall-clock timestamps for this tight creation loop; tied times
    # deliberately sort by record ID, not ledger insertion order.
    start = datetime(2026, 10, 20, tzinfo=timezone.utc)
    records = [replace(record, created_at=(start + timedelta(milliseconds=offset + index)).isoformat())
               for index, record in enumerate(records)]
    view.session.measurements.update((record.id, record) for record in records)
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
    # Accepted data and the compatibility models remain synchronous. The
    # secondary dock presentation is coalesced into the next UI event turn.
    assert workspace.current_model.ids == workspace.history_model.ids == ids
    assert list(workspace.session.measurements) == ids
    assert workspace._records_notification_timer.isActive()
    settle()
    assert presenter.model.keys == [("measurement", key) for key in ids]
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


def test_opening_records_before_queued_publication_reads_current_ledger(workspace):
    toggle_records(workspace)
    presenter = workspace._records_presenter()
    toggle_records(workspace)
    table = presenter.table
    table.scrollToBottom = Mock(wraps=table.scrollToBottom)
    ids = add_records(workspace, 60)
    assert workspace._records_notification_timer.isActive()
    # Open immediately, before any processEvents: opening synchronizes from
    # the accepted ledger and does not depend on the queued notification.
    QTest.mouseClick(workspace.records_toggle, Qt.MouseButton.LeftButton)
    assert presenter.model.keys == [("measurement", key) for key in ids]
    settle()
    table.scrollToBottom.assert_called_once()
    assert table.verticalScrollBar().value() == table.verticalScrollBar().maximum()


def test_queued_publication_cannot_replace_a_newly_bound_trial(workspace):
    toggle_records(workspace)
    presenter = workspace._records_presenter()
    ids = add_records(workspace, 3)
    assert workspace._records_notification_timer.isActive()
    session = make_session()
    other_record = session.add_measurement(41, "cashmere", field_id="other-field")
    other = SimpleNamespace(session=session, controller=SimpleNamespace(field_id="other-field"))
    presenter.bind(other)
    settle()
    assert presenter.model.session is session
    assert presenter.model.keys == [("measurement", other_record.id)]
    # Returning binds the original ledger, so ignored old-view notifications
    # never mean accepted rows were lost.
    presenter.bind(workspace)
    settle()
    assert presenter.model.keys == [("measurement", key) for key in ids]


def test_closed_workspace_ignores_queued_secondary_updates(workspace, monkeypatch):
    toggle_records(workspace)
    ids = add_records(workspace, 3)
    assert workspace._records_notification_timer.isActive()
    notify = Mock()
    monkeypatch.setattr(workspace, "_notify_records", notify)
    # Keep the Python fixture's widget alive for its own teardown while using
    # the real close path, including controller and presenter shutdown.
    workspace.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
    assert workspace.close()
    assert workspace._closed
    notify.reset_mock()
    settle()
    notify.assert_not_called()
    assert list(workspace.session.measurements) == ids


def test_queued_publication_uses_deterministic_id_order_for_equal_timestamps(workspace):
    toggle_records(workspace)
    presenter = workspace._records_presenter()
    records = []
    for record_id in ("z-record", "a-record", "m-record"):
        record = workspace.session.add_measurement(20, "wool", id=record_id,
            field_id=workspace.controller.field_id)
        record = replace(record, created_at="2026-10-20T00:00:00+00:00")
        workspace.session.measurements[record.id] = record
        records.append(record)
    workspace.controller.current_records.extend(record.id for record in records)
    workspace._records_accepted(records)
    assert workspace.current_model.ids == ["z-record", "a-record", "m-record"]
    settle()
    assert presenter.model.keys == [("measurement", key) for key in ("a-record", "m-record", "z-record")]
    assert presenter.table.verticalScrollBar().value() == presenter.table.verticalScrollBar().maximum()


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
