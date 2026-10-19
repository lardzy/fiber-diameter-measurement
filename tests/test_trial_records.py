"""Trial record placement and presentation must not mutate either ledger."""
from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget

from fdm.live_experiment import ExperimentSession, default_materials
from fdm.ui.trial_records import (
    AcquisitionDockHost, KEY_ROLE, RecordsPresentationController, TrialRecordsPanel,
    TrialTimelineFilter, TrialTimelineModel,
)


def session():
    value = ExperimentSession(save_images=False)
    value.add_operator("甲")
    for direction in ("warp", "weft"):
        value.set_materials(default_materials(), direction=direction)
    return value


def events():
    for _ in range(3):
        QApplication.processEvents()


def test_timeline_merges_signed_count_without_fabricated_measurement():
    data = session()
    first = data.add_measurement(21.25, "wool", id="first", field_id="one")
    addition = data.adjust_count("wool", 3, field_id="one")
    reduction = data.adjust_count("wool", -1, field_id="two", note="重新核对")
    last = data.add_measurement(22, "wool", id="last", field_id="two")
    # Windows may assign the same clock tick to consecutive ledger events.
    # This case tests chronological merge; the following case covers ties.
    for second, (ledger, event) in enumerate(((data.measurements, first),
            (data.count_adjustments, addition), (data.count_adjustments, reduction),
            (data.measurements, last))):
        ledger[event.id] = replace(event, created_at=f"2026-10-19T00:00:0{second}+00:00")
    before = data.to_dict()
    model = TrialTimelineModel(data)
    assert model.keys == [("measurement", first.id), ("count", addition.id),
                          ("count", reduction.id), ("measurement", last.id)]
    assert model.data(model.index(0, 2)) == "21.2500 μm"
    assert model.data(model.index(1, 2)) == "+3 根"
    assert model.data(model.index(2, 2)) == "-1 根"
    assert model.data(model.index(1, 8)) == "不适用"
    assert model.data(model.index(2, 0), KEY_ROLE) == ("count", reduction.id)
    assert "重新核对" in model.data(model.index(2, 0), Qt.ItemDataRole.ToolTipRole)
    assert data.to_dict() == before


def test_timestamp_utc_naive_invalid_and_equal_ties_are_stable_without_migration():
    data = session()
    a = data.add_measurement(20, "wool", id="z")
    b = data.add_measurement(21, "wool", id="a")
    c = data.add_measurement(22, "wool", id="bad")
    adjustment = data.adjust_count("wool", 2)
    data.measurements[a.id] = replace(a, created_at="2026-10-19T08:00:00+08:00")
    data.measurements[b.id] = replace(b, created_at="2026-10-19T00:00:00")
    data.measurements[c.id] = replace(c, created_at="unavailable")
    data.count_adjustments[adjustment.id] = replace(adjustment, created_at="2026-10-19T00:00:00Z")
    original = data.to_dict()
    keys = [("measurement", "bad"), ("count", adjustment.id), ("measurement", "a"), ("measurement", "z")]
    assert TrialTimelineModel(data).keys == keys
    reloaded = ExperimentSession.from_dict(original)
    assert TrialTimelineModel(reloaded).keys == keys
    assert data.to_dict() == original
    assert TrialTimelineModel(data).data(TrialTimelineModel(data).index(0, 6)) == "时间未知"


def test_explicit_publication_is_incremental_even_when_notifications_share_a_revision():
    data = session()
    model = TrialTimelineModel(data)
    model._all_keys = Mock(side_effect=AssertionError("incremental publication scanned the ledger"))
    a = data.add_measurement(20, "wool")
    b = data.adjust_count("wool", 2)
    model.sync([("measurement", a.id)])
    model.sync([("count", b.id)])
    assert len(model.keys) == 2
    model.sync()
    model._all_keys.assert_not_called()
    data.edit_measurement(a.id, diameter_um=23)
    model.sync([("measurement", a.id)])
    assert len(model.keys) == 2
    assert model.data(model.index(model.rows_by_key[("measurement", a.id)], 2)) == "23.0000 μm"
    data.delete_measurement(a.id)
    model.sync([("measurement", a.id)])
    assert model.keys == [("count", b.id)]


def test_undo_redo_rebuild_keeps_original_order_and_signed_count_balance():
    data = session()
    a = data.add_measurement(20, "wool")
    b = data.adjust_count("wool", 5)
    c = data.adjust_count("wool", -2, note=f"修正计数记录 {b.id}")
    model = TrialTimelineModel(data)
    original = list(model.keys)
    data.undo()
    model.sync(rebuild=True)
    assert ("count", c.id) not in model.keys
    data.redo()
    model.sync(rebuild=True)
    assert model.keys == original
    assert sum(event.delta for event in data.count_adjustments.values()) == 3
    data.delete_measurement(a.id)
    model.sync(rebuild=True)
    assert model.rowCount() == 2


def test_confirmed_batch_publishes_one_contiguous_table_change():
    data = session()
    model = TrialTimelineModel(data)
    records = [data.add_measurement(20 + index / 1000, "wool") for index in range(2000)]
    inserted = QSignalSpy(model.rowsInserted)
    model.sync([("measurement", record.id) for record in records])
    assert model.rowCount() == 2000
    assert inserted.count() == 1
    assert inserted.at(0)[1:] == [0, 1999]


def test_large_ledger_idle_notifications_do_not_scan_and_evidence_only_updates_one_row():
    data = session()
    for index in range(20000):
        data.add_measurement(20 + (index % 100) / 1000, "wool")
    model = TrialTimelineModel(data)
    all_keys = model._all_keys
    model._all_keys = Mock(side_effect=AssertionError("an unchanged revision scanned the ledger"))
    model._order = Mock(wraps=model._order)
    for _ in range(200):
        model.sync()
    model._all_keys.assert_not_called()
    model._order.assert_not_called()
    model._all_keys = Mock(wraps=all_keys)
    changed = QSignalSpy(model.dataChanged)
    reset = QSignalSpy(model.modelReset)
    key = model.keys[12345]
    data.set_evidence(key[1], {"state": "saved", "path": "evidence.png"})
    model.sync()
    assert model._all_keys.call_count == 1
    assert model._order.call_count == 1
    assert changed.count() == 1 and reset.count() == 0
    assert changed.at(0)[0].row() == 12345
    assert model.data(model.index(12345, 8)) == "已保存"


def test_metadata_change_refreshes_labels_without_reordering_records():
    data = session()
    for _ in range(10):
        data.add_measurement(20, "wool")
    model = TrialTimelineModel(data)
    model._order = Mock(side_effect=AssertionError("renaming an operator sorted the ledger"))
    changed = QSignalSpy(model.dataChanged)
    data.rename_operator("改名后")
    model.sync()
    assert changed.count() == 1
    assert model.data(model.index(0, 4)) == "改名后"
    model._order.assert_not_called()


def test_filters_cover_three_operators_directions_types_and_dynamic_field():
    data = session()
    operators = [data.active_operator_id, data.add_operator("乙"), data.add_operator("丙")]
    for operator in operators:
        for direction in ("warp", "weft"):
            data.add_measurement(20, "wool", operator_id=operator, direction=direction, field_id="old")
            data.adjust_count("wool", 1, operator_id=operator, direction=direction, field_id="new")
    model = TrialTimelineModel(data)
    proxy = TrialTimelineFilter()
    proxy.setSourceModel(model)
    assert proxy.rowCount() == 12
    proxy.apply(operator_id=operators[1], direction="weft")
    assert proxy.rowCount() == 2
    proxy.apply(record_type="count", current_field_only=True, field_id="new")
    assert proxy.rowCount() == 6
    proxy.apply(current_field_only=True, field_id="old")
    assert proxy.rowCount() == 6
    proxy.apply(current_field_only=True, field_id="absent")
    assert proxy.rowCount() == 0
    proxy.apply()
    assert proxy.rowCount() == 12


@pytest.fixture
def panel():
    value = TrialRecordsPanel()
    value.resize(470, 350)
    yield value
    value.close()
    value.deleteLater()
    events()


def test_panel_open_to_latest_then_respects_reading_position_and_selection(panel):
    data = session()
    for i in range(100):
        data.add_measurement(20 + i / 100, "wool")
    panel.set_session(data)
    panel.show()
    events()
    bar = panel.table.verticalScrollBar()
    assert bar.maximum() > 0 and bar.value() == bar.maximum()
    key = panel.model.keys[3]
    panel.select_key(key)
    bar.setValue(0)
    assert not panel.following
    added = data.adjust_count("wool", 1)
    panel.sync([("count", added.id)])
    events()
    assert bar.value() == 0
    assert tuple(panel.selected_key()) == key
    assert panel.latest_button.text() == "新增 1 条 · 回到最新"
    panel.latest_button.click()
    events()
    assert panel.following and bar.value() == bar.maximum()
    assert not panel.latest_button.isEnabled()
    assert tuple(panel.selected_key()) == key


def test_hidden_panel_defers_scrolling_but_keeps_both_types_current(panel):
    data = session()
    panel.set_session(data)
    panel.table.scrollToBottom = Mock(wraps=panel.table.scrollToBottom)
    record = data.add_measurement(20, "wool")
    count = data.adjust_count("wool", 1)
    panel.sync([("measurement", record.id), ("count", count.id)])
    panel.table.scrollToBottom.assert_not_called()
    assert panel.model.rowCount() == 2
    panel.show()
    events()
    panel.table.scrollToBottom.assert_called_once()


def test_open_reset_filters_and_count_actions_are_typed(panel):
    data = session()
    measurement = data.add_measurement(20, "wool")
    count = data.adjust_count("wool", 1)
    panel.set_session(data, "field")
    panel.type_filter.setCurrentIndex(1)
    assert panel.proxy.rowCount() == 1
    panel.reset_filters()
    assert panel.proxy.rowCount() == 2
    requested = []
    panel.actionRequested.connect(lambda action, key: requested.append((action, tuple(key))))
    panel.select_key(("count", count.id))
    assert panel.action_buttons["correct_count"].isEnabled()
    assert all(not panel.action_buttons[action].isEnabled() for action in ("edit", "delete", "evidence"))
    panel.action_buttons["correct_count"].click()
    assert requested == [("correct_count", ("count", count.id))]
    panel.select_key(("measurement", measurement.id))
    assert not panel.action_buttons["correct_count"].isEnabled()
    assert panel.action_buttons["edit"].isEnabled()


def test_empty_unconfigured_trial_does_not_create_operator_or_categories(panel):
    data = ExperimentSession()
    before = data.to_dict()
    panel.set_session(data)
    panel.reset_filters()
    panel.sync()
    assert panel.model.rowCount() == 0
    assert panel.operator_filter.count() == 1
    assert not any(button.isEnabled() for button in panel.action_buttons.values())
    assert data.to_dict() == before
    data.add_operator("明确填写的操作者")
    panel.sync()
    assert panel.operator_filter.count() == 2
    assert not data.operators[data.active_operator_id].materials["warp"]


@pytest.fixture
def presenter(tmp_path):
    window = QMainWindow()
    root = QWidget(window)
    layout = QVBoxLayout(root)
    navigation = [QLabel("主导航", root), QLabel("采集子导航", root)]
    for item in navigation:
        item.setFixedHeight(36)
        layout.addWidget(item)
    host = AcquisitionDockHost(root)
    host.setCentralWidget(QLabel("图像与操作台", host))
    layout.addWidget(host, 1)
    window.setCentralWidget(root)
    state = {"current": None}
    actions = []
    controller = RecordsPresentationController(host, is_current=lambda workspace: state["current"] is workspace,
        action_callback=lambda workspace, action, key: actions.append((workspace, action, key)),
        preferences_path=tmp_path / "records.json")
    workspace = SimpleNamespace(session=session(), controller=SimpleNamespace(field_id="initial"))
    state["current"] = workspace
    controller.bind(workspace)
    window.resize(1600, 900)
    window.show()
    events()
    yield controller, workspace, window, navigation, state, actions
    controller.shutdown()
    window.close()
    window.deleteLater()
    events()


def rect(widget):
    value = widget.geometry()
    return value.x(), value.y(), value.width(), value.height()


def test_dock_default_left_open_and_moves_leave_both_navigations_fixed(presenter):
    controller, workspace, window, navigation, _, _ = presenter
    assert not controller.host.isWindow()
    assert controller.preferences["position"] == "left"
    assert not controller.is_open
    assert controller.open()
    events()
    assert controller.host.dockWidgetArea(controller.dock) == Qt.DockWidgetArea.LeftDockWidgetArea
    original = [rect(item) for item in navigation]
    body = rect(controller.host)
    for position in ("bottom", "floating", "left") * 3:
        controller.set_position(position)
        events()
        assert [rect(item) for item in navigation] == original
        assert rect(controller.host) == body
    controller.set_position("floating")
    assert controller.dock.parentWidget() is controller.host
    assert controller.dock.windowType() == Qt.WindowType.Tool
    assert not controller.dock.windowFlags() & Qt.WindowType.WindowStaysOnTopHint


def test_programmatic_hide_does_not_save_visibility_or_dirty_trial(presenter):
    controller, workspace, _, _, _, _ = presenter
    before = workspace.session.to_dict()
    controller.open()
    controller.set_position("bottom")
    events()
    controller.flush_preferences()
    persisted = controller.preferences_path.read_bytes()
    controller.suspend()
    events()
    assert not controller.is_open
    assert controller.preferences["visible"] is True
    assert controller.preferences_path.read_bytes() == persisted
    assert not controller._save_timer.isActive()
    assert workspace.session.to_dict() == before
    controller.bind(workspace)
    events()
    assert controller.is_open
    controller.close()
    controller.flush_preferences()
    assert json.loads(controller.preferences_path.read_text())["visible"] is False


def test_action_routing_requires_active_current_trial_and_valid_typed_key(presenter):
    controller, workspace, _, _, state, actions = presenter
    event = workspace.session.adjust_count("wool", 2)
    controller.notify_change(workspace, [("count", event.id)])
    controller.open()
    controller.panel.select_key(("count", event.id))
    controller.panel.action_buttons["correct_count"].click()
    assert actions == [(workspace, "correct_count", ("count", event.id))]
    state["current"] = None
    controller.panel.action_buttons["correct_count"].click()
    assert len(actions) == 1
    state["current"] = workspace
    controller.suspend()
    controller._action("correct_count", ("count", event.id))
    assert len(actions) == 1
    assert controller.contains_focus(controller.panel.operator_filter)
    assert controller.contains_focus(controller.table.viewport())
    assert not controller.contains_focus(controller.host.centralWidget())


def test_float_suspend_and_shutdown_cannot_reactivate_old_trial(presenter):
    controller, workspace, _, _, _, actions = presenter
    record = workspace.session.add_measurement(20, "wool")
    controller.notify_change(workspace, [("measurement", record.id)])
    controller.open()
    controller.set_position("floating")
    events()
    controller.flush_preferences()
    saved = controller.preferences_path.read_bytes()
    controller.suspend()
    assert not controller.dock.isVisible()
    assert controller.preferences["position"] == "floating"
    assert controller.preferences["visible"] is True
    events()
    assert controller.preferences_path.read_bytes() == saved
    controller.shutdown()
    controller.bind(workspace)
    assert not controller.open(workspace)
    controller._action("edit", ("measurement", record.id))
    assert not actions
    assert controller.model.session is None


def test_trial_rebinding_disconnects_old_data_and_restores_new_metadata(presenter):
    controller, old, _, _, state, _ = presenter
    controller.open()
    old.session.adjust_count("wool", 2)
    controller.notify_change(old)
    new = SimpleNamespace(session=session(), controller=SimpleNamespace(field_id="new-field"))
    new.session.sample_name = "新样品"
    state["current"] = new
    controller.bind(new)
    assert controller.model.session is new.session and controller.model.rowCount() == 0
    assert "新样品" in controller.dock.windowTitle()
    old.session.adjust_count("wool", 1)
    controller.notify_change(old)
    assert controller.model.rowCount() == 0
    event = new.session.adjust_count("wool", 1, field_id="new-field")
    controller.notify_change(new, [("count", event.id)])
    assert controller.model.keys == [("count", event.id)]
    new.controller.field_id = "next-field"
    controller.notify_change(new)
    controller.panel.field_filter.setCurrentIndex(1)
    assert controller.proxy.rowCount() == 0


def test_preferences_restore_columns_float_geometry_and_reset_layout(presenter, tmp_path):
    controller, workspace, _, _, _, _ = presenter
    controller.open()
    controller.table.setColumnWidth(2, 150)
    controller.set_position("floating")
    events()
    controller.flush_preferences()
    persisted = json.loads(controller.preferences_path.read_text())
    assert persisted["columns"][2] == 150
    # Simulate the monitor that hosted the tool window being disconnected.
    persisted["floating_geometry"] = [100000, 100000, 900, 600]
    path = tmp_path / "restored.json"
    path.write_text(json.dumps(persisted))
    host = AcquisitionDockHost()
    host.setCentralWidget(QWidget(host))
    other = RecordsPresentationController(host, preferences_path=path)
    try:
        other.bind(workspace)
        other.open()
        events()
        assert other.dock.isFloating()
        assert other.table.columnWidth(2) == 150
        assert any(screen.availableGeometry().contains(other.dock.geometry()) for screen in QApplication.screens())
        other.reset_layout()
        assert other.preferences["position"] == "left"
        assert not other.dock.isFloating()
        assert other.table.columnWidth(2) == TrialRecordsPanel.DEFAULT_COLUMNS[2]
    finally:
        other.shutdown()
        host.close()
        host.deleteLater()
