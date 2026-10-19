"""Exercise the shared records dock through the real acquisition navigation.

The fake camera supplies frames, and domain calls seed historical records. All
page transitions, record opening, corrections and keyboard input use the same
widgets and actions as the application, rather than assigning page state.
"""
from pathlib import Path
import sys

from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication, QDockWidget, QPushButton
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/validation"))
from validate_workflow_redesign import WorkflowHarness, settle

from fdm.live_experiment import default_materials
from fdm.ui.trial_records import KEY_ROLE, TrialTimelineModel


@pytest.fixture
def harness(tmp_path):
    with WorkflowHarness(tmp_path / "profile", size=(1600, 900)) as value:
        value.enter()
        yield value
        assert not value.slot_errors, "\n".join(value.slot_errors)
        assert not value.modal_messages


def content_page(harness):
    page = harness.select(1)
    # Empty entry is intentional: test data must explicitly configure fibers.
    assert page.material.count() == 0
    for direction in ("warp", "weft"):
        page.session.set_materials(default_materials(), direction=direction)
    page._refresh_materials()
    page._changed()
    return page


def seed_history(page, rows_per_scope=15):
    session = page.session
    session.rename_operator("甲")
    operators = [session.active_operator_id, session.add_operator("乙"), session.add_operator("丙")]
    for operator in operators:
        for direction in ("warp", "weft"):
            session.set_materials(default_materials(), operator_id=operator, direction=direction)
            for row in range(rows_per_scope):
                session.add_measurement(20 + row / 10, "wool", operator_id=operator,
                    direction=direction, field_id=f"past-{operator}-{direction}")
            session.adjust_count("wool", 3, operator_id=operator, direction=direction,
                field_id=f"past-{operator}-{direction}")
    page._refresh_operators()
    page._rebuild_records()
    settle()
    return operators


def click(widget):
    assert widget.isVisible() and widget.isEnabled()
    QTest.mouseClick(widget, Qt.MouseButton.LeftButton)
    settle()


def open_records(harness, page):
    presenter = harness.hub.records
    assert not presenter.is_open
    click(page.records_toggle)
    assert presenter.is_open and presenter.dock.isVisible()
    assert page.records_toggle.isChecked()
    return presenter


def choose_position(presenter, name):
    labels = {"left": "停靠左侧", "bottom": "停靠底部", "floating": "浮动窗口"}
    action = next(action for action in presenter.panel.position_button.menu().actions()
                  if action.text() == labels[name])
    action.trigger()
    settle()
    assert presenter.preferences["position"] == name


def navigation_geometry(harness):
    widgets = [*harness.window._workspace_buttons, harness.hub.navigation, harness.hub.capture_button]
    return tuple(QRect(widget.mapTo(harness.window, QPoint()), widget.size()) for widget in widgets)


def test_record_button_toggles_and_reopening_resets_all_scopes_to_latest(harness):
    page = content_page(harness)
    operators = seed_history(page)
    presenter = open_records(harness, page)
    panel = presenter.panel
    expected = 3 * 2 * 16
    assert panel.proxy.rowCount() == expected
    assert {record.operator_id for key in panel.model.keys
            if (record := panel.model.record(key))} == set(operators)
    assert {panel.model.record(key).direction for key in panel.model.keys} == {"warp", "weft"}
    assert {key[0] for key in panel.model.keys} == {"measurement", "count"}
    bar = panel.table.verticalScrollBar()
    assert bar.maximum() > 0 and bar.value() == bar.maximum()
    panel.operator_filter.setCurrentIndex(2)
    panel.direction_filter.setCurrentIndex(2)
    panel.type_filter.setCurrentIndex(1)
    settle()
    assert panel.proxy.rowCount() == 15
    click(page.records_toggle)
    assert not presenter.is_open and not page.records_toggle.isChecked()
    click(page.records_toggle)
    assert panel.proxy.rowCount() == expected
    assert all(widget.currentIndex() == 0 for widget in (panel.operator_filter,
        panel.direction_filter, panel.type_filter, panel.field_filter))
    assert bar.value() == bar.maximum()
    assert panel.table.indexAt(QPoint(5, panel.table.viewport().height() - 5)).isValid()


def test_new_record_respects_upward_reading_then_explicit_latest_resumes_follow(harness):
    page = content_page(harness)
    seed_history(page)
    presenter = open_records(harness, page)
    panel, table = presenter.panel, presenter.table
    # Home is a user action on the focused table, and must pause follow mode.
    table.setFocus()
    QTest.keyClick(table, Qt.Key.Key_Home, Qt.KeyboardModifier.ControlModifier)
    settle()
    bar = table.verticalScrollBar()
    assert bar.value() == 0 and not panel.following
    selected = tuple(panel.selected_key())
    first_visible = table.indexAt(QPoint(5, 5)).data(KEY_ROLE)
    added = page.session.add_measurement(23.5, "wool", field_id=page.controller.field_id)
    page._record_accepted(added)
    page._changed()
    settle()
    assert bar.value() == 0
    assert table.indexAt(QPoint(5, 5)).data(KEY_ROLE) == first_visible
    assert tuple(panel.selected_key()) == selected
    assert panel.latest_button.text() == "新增 1 条 · 回到最新"
    click(panel.latest_button)
    assert panel.following and bar.value() == bar.maximum()
    assert not panel.latest_button.isEnabled()
    next_record = page.session.add_measurement(24, "wool", field_id=page.controller.field_id)
    page._record_accepted(next_record)
    page._changed()
    settle()
    assert bar.value() == bar.maximum()
    assert tuple(panel.selected_key()) == selected


@pytest.mark.parametrize("position", ["left", "bottom", "floating"])
def test_position_menu_keeps_navigation_fixed_and_return_preserves_visibility(harness, position):
    page = content_page(harness)
    seed_history(page, rows_per_scope=2)
    baseline = navigation_geometry(harness)
    presenter = open_records(harness, page)
    choose_position(presenter, position)
    assert navigation_geometry(harness) == baseline
    harness.assert_navigation()
    assert presenter.dock.isFloating() is (position == "floating")
    if position == "floating":
        assert presenter.dock.parentWidget() is harness.hub.body_host
        assert presenter.dock.windowType() == Qt.WindowType.Tool
        assert not (presenter.dock.windowFlags() & Qt.WindowType.WindowStaysOnTopHint)
    else:
        assert harness.hub.body_host.dockWidgetArea(presenter.dock) == (
            Qt.DockWidgetArea.LeftDockWidgetArea if position == "left" else Qt.DockWidgetArea.BottomDockWidgetArea)
    presenter.flush_preferences()
    before = presenter.preferences_path.read_bytes()
    # Use the explicit exit, then the actual content navigation tab.
    click(page.return_capture_button)
    assert harness.hub.pages.currentWidget() is harness.hub.capture_page
    assert not presenter.is_open and presenter.dock.isHidden()
    assert not page.records_toggle.isChecked()
    assert presenter.preferences_path.read_bytes() == before
    assert harness.select(1) is page
    assert presenter.is_open and page.records_toggle.isChecked()
    assert presenter.preferences["position"] == position
    assert navigation_geometry(harness) == baseline
    harness.leave()
    assert not presenter.is_open and presenter.dock.isHidden()
    harness.enter()
    assert presenter.workspace is page and presenter.is_open
    assert page.records_toggle.isChecked()
    assert navigation_geometry(harness) == baseline


def test_signed_correction_uses_selected_original_scope_and_undo_is_visible(harness, monkeypatch):
    import fdm.ui.live_experiment_workspace as workspaces
    page = content_page(harness)
    operators = seed_history(page, rows_per_scope=2)
    presenter = open_records(harness, page)
    original = next(event for event in page.session.count_adjustments.values()
                    if event.operator_id == operators[0] and event.direction == "weft")
    assert page.session.active_operator_id != original.operator_id
    assert page.session.active_direction != original.direction
    assert page.controller.field_id != original.field_id
    presenter.panel.select_key(("count", original.id))
    actions = presenter.panel.action_buttons
    assert actions["correct_count"].isEnabled()
    assert not any(actions[name].isEnabled() for name in ("edit", "delete", "evidence"))
    prompts = []
    monkeypatch.setattr(workspaces.QInputDialog, "getInt", lambda *args, **kwargs:
                        prompts.append((args, kwargs)) or (-2, True))
    measurements = dict(page.session.measurements)
    keys_before = set(page.session.count_adjustments)
    click(actions["correct_count"])
    new_keys = set(page.session.count_adjustments) - keys_before
    assert len(new_keys) == 1
    correction = page.session.count_adjustments[new_keys.pop()]
    assert correction.delta == -2
    assert (correction.operator_id, correction.direction, correction.material_id, correction.field_id) == (
        original.operator_id, original.direction, original.material_id, original.field_id)
    assert page.session.count_adjustments[original.id] is original
    assert original.id in correction.note
    assert page.session.measurements == measurements
    assert "甲／纬向" in prompts[0][0][2]
    assert prompts[0][0][4] == -3
    assert ("count", correction.id) in presenter.model.rows_by_key
    undo = next(button for button in page.findChildren(QPushButton) if button.accessibleName() == "撤销")
    click(undo)
    assert correction.id not in page.session.count_adjustments
    assert ("count", correction.id) not in presenter.model.rows_by_key
    assert tuple(presenter.panel.selected_key()) == ("count", original.id)
    assert page.session.measurements == measurements


@pytest.mark.parametrize("floating", [False, True])
def test_digits_in_records_do_not_measure_or_count_even_with_count_mode_enabled(harness, floating):
    page = content_page(harness)
    seed_history(page, rows_per_scope=2)
    page.keyboard_mode.setCurrentIndex(1)
    harness.window.activateWindow()
    page.canvas.setFocus()
    settle()
    count_before = len(page.session.count_adjustments)
    QTest.keyClick(page.canvas, Qt.Key.Key_1)
    settle()
    assert len(page.session.count_adjustments) == count_before + 1
    presenter = open_records(harness, page)
    if floating:
        choose_position(presenter, "floating")
        presenter.dock.activateWindow()
    else:
        harness.window.activateWindow()
    settle()
    table = presenter.table
    click(table.viewport())
    table.setFocus()
    settle()
    assert presenter.contains_focus(QApplication.focusWidget())
    before = page.session.to_dict()
    for key in (Qt.Key.Key_1, Qt.Key.Key_2, Qt.Key.Key_Minus):
        QTest.keyClick(table, key)
    settle()
    assert page.session.to_dict() == before
    assert page.controller.draft is None
    # A delayed action from a suspended records view must also be rejected.
    count_key = next(key for key in presenter.model.keys if key[0] == "count")
    harness.select(0)
    presenter.panel.actionRequested.emit("correct_count", count_key)
    assert page.session.to_dict() == before
    diameter = harness.select(2)
    harness.window.activateWindow()
    diameter.canvas.setFocus()
    settle()
    QTest.keyClick(diameter.canvas, Qt.Key.Key_1)
    settle()
    assert page.session.to_dict() == before
    assert not diameter.session.count_adjustments


def test_repeated_real_navigation_reuses_one_model_dock_and_capture_connection(harness):
    content = content_page(harness)
    seed_history(content, rows_per_scope=1)
    presenter = open_records(harness, content)
    model, panel, dock = presenter.model, presenter.panel, presenter.dock
    starts = harness.backend.starts
    for index in (0, 1, 2, 1) * 50:
        page = harness.select(index)
        assert presenter.model is model and presenter.panel is panel and presenter.dock is dock
        assert harness.hub.body_host.findChildren(QDockWidget) == [dock]
        assert harness.hub.body_host.findChildren(TrialTimelineModel) == [model]
        assert presenter.is_open == (index != 0)
        if index:
            assert presenter.model.session is page.session
            assert page.records_toggle.isChecked()
        harness.assert_navigation()
    assert harness.backend.starts == starts
    assert not [event for event in harness.observer.events
                if event["class"] not in {"MainWindow", "QTipLabel"}]
    # Frame refreshes are presentation-only; they must not rebuild the ledger.
    resets = QSignalSpy(model.modelReset)
    rows = model.rowCount()
    for _ in range(100):
        harness.backend.tick()
        settle(rounds=1)
    assert resets.count() == 0 and model.rowCount() == rows


@pytest.mark.parametrize("size,minimum_rows", [((1067, 600), 2), ((1366, 768), 4), ((1600, 900), 4)])
def test_bottom_dock_fits_small_logical_windows_and_preserves_latest(tmp_path, size, minimum_rows):
    with WorkflowHarness(tmp_path / "profile", size=size) as harness:
        harness.enter()
        page = content_page(harness)
        seed_history(page)
        presenter = open_records(harness, page)
        baseline = navigation_geometry(harness)
        for position in ("bottom", "left", "bottom", "floating", "bottom"):
            choose_position(presenter, position)
            # LayoutRequest propagation and the trial's 80 ms refresh can
            # reveal minimum-size growth after the first settled frame.
            QTest.qWait(200)
            harness.assert_size()
            assert navigation_geometry(harness) == baseline
            bar = presenter.table.verticalScrollBar()
            assert presenter.panel.following and bar.value() == bar.maximum()
            assert page.save_button.isVisible() and page.export_button.isVisible()
            for button in (page.save_button, page.export_button):
                rect = QRect(button.mapTo(harness.window, QPoint()), button.size())
                assert harness.window.rect().contains(rect)
            if position == "bottom":
                assert presenter.table.viewport().height() >= minimum_rows * presenter.table.verticalHeader().defaultSectionSize()
                assert len({widget.y() for widget in (presenter.panel.operator_filter,
                    presenter.panel.direction_filter, presenter.panel.type_filter, presenter.panel.field_filter)}) == 1
        # A user reading older data is not pulled to the bottom when docking.
        presenter.table.setFocus()
        QTest.keyClick(presenter.table, Qt.Key.Key_Home, Qt.KeyboardModifier.ControlModifier)
        settle()
        assert not presenter.panel.following
        choose_position(presenter, "left")
        choose_position(presenter, "bottom")
        assert not presenter.panel.following
        assert presenter.table.verticalScrollBar().value() == 0
        assert not harness.slot_errors and not harness.modal_messages


def test_bottom_dock_resize_limits_actual_height_without_losing_preferred_height(harness):
    page = content_page(harness)
    seed_history(page)
    presenter = open_records(harness, page)
    choose_position(presenter, "bottom")
    preferred = presenter.preferences["bottom_height"]
    assert preferred >= 250
    harness.size = (1067, 600)
    harness.window.resize(*harness.size)
    QTest.qWait(200)
    settle()
    harness.assert_size()
    assert presenter.dock.height() < preferred
    assert presenter.preferences["bottom_height"] == preferred
    assert presenter.table.verticalScrollBar().value() == presenter.table.verticalScrollBar().maximum()
    harness.size = (1600, 900)
    harness.window.resize(*harness.size)
    QTest.qWait(200)
    settle()
    harness.assert_size()
    assert presenter.dock.height() == preferred
