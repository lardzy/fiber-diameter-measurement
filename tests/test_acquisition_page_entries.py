"""Explicit page actions stay distinct from camera session commands."""
import json

import pytest
from PySide6.QtWidgets import QApplication, QToolButton

from fdm.ui.main_window import MainWindow
from acquisition_test_support import wait_workbench_ready


@pytest.fixture
def window(monkeypatch):
    monkeypatch.setattr('fdm.services.capture.available_capture_backends', lambda: [])
    host = MainWindow()
    monkeypatch.setattr(host, 'start_live_preview', lambda **kwargs: None)
    host.resize(1280, 800)
    host.show()
    QApplication.processEvents()
    yield host
    monkeypatch.setattr(host, '_confirm_close_documents_for_transition', lambda *_: True)
    host.close()
    QApplication.processEvents()


def enter(window, index):
    window._acquisition_page_actions[index].trigger()
    hub = window._acquisition_workbench
    wait_workbench_ready(hub)
    return hub


def test_three_page_actions_are_shared_by_split_menu_trial_menu_and_search(window):
    actions = window._acquisition_page_actions
    assert [a.text() for a in actions] == ['预览与抓拍', '含量试验', '直径试验']
    button = window._acquisition_workspace_button
    assert button.popupMode() == QToolButton.ToolButtonPopupMode.MenuButtonPopup
    assert button.defaultAction() is window.acquisition_workspace_action
    assert button.menu().actions() == list(actions)
    menu_actions = window.menuBar().actions()
    menu = next(a for a in menu_actions if a.text() == '试验').menu()
    assert menu.actions()[:3] == list(actions)
    entries = {a: (path, keywords) for a, path, keywords in window._command_search_entries()}
    for action in actions:
        assert action in entries and entries[action][0] == '试验'
    assert '实时' in entries[actions[0]][1]
    assert window._snapshot_controls.view.text() == '查看刚拍图像'


def test_main_entry_ignores_saved_last_trial_and_does_not_create_trial_or_excel(window, monkeypatch):
    from fdm.ui.acquisition_workbench import AcquisitionWorkbench
    path = AcquisitionWorkbench._preferences_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'version': 2, 'page': 1, 'sidebar_width': 470}))
    calls = []
    monkeypatch.setattr('fdm.ui.trial_excel_lifecycle.ensure_live_excel',
                        lambda *_: calls.append('excel'))
    window.acquisition_workspace_action.trigger()
    hub = window._acquisition_workbench
    assert hub._selected == 0 and hub.navigation.currentIndex() == 0
    assert hub.pages.currentWidget() is hub.capture_page
    QApplication.processEvents()
    assert not getattr(window, '_live_experiment_workspaces', {})
    assert not window.project.live_experiments and not calls
    assert hub.navigation.tabText(0) == '预览与抓拍'


def test_explicit_actions_preserve_trial_identity_parameters_and_excel_owner(window, monkeypatch):
    hub = enter(window, 1)
    content = hub._current_trial()
    content.sample.setText('入口回归样品')
    content.commit_pending_edits()
    from fdm.live_experiment import default_materials
    content.session.set_materials(default_materials())
    record = content.session.add_measurement(18.5, 'wool', field_id=content.controller.field_id)
    session, coordinator = content.session, content.live_excel
    # A running camera cannot short-circuit the requested page navigation.
    monkeypatch.setattr(window._capture_manager, 'is_preview_active', lambda: True)
    for _ in range(3):
        window.acquisition_workspace_action.trigger()
        assert hub.pages.currentWidget() is hub.capture_page
        assert hub._selected == 0
        enter(window, 2)
        diameter = hub._current_trial()
        assert diameter.session.workspace_options['mode'] == 'diameter'
        enter(window, 1)
        assert hub._current_trial() is content
        assert content.session is session and record.id in session.measurements
        assert content.session.sample_name == '入口回归样品'
        assert content.live_excel is coordinator
    monkeypatch.setattr(window._capture_manager, 'is_preview_active', lambda: False)


def test_rejected_page_selection_preserves_page_layout_and_workspace_state(window, monkeypatch):
    hub = enter(window, 1)
    content = hub._current_trial()
    monkeypatch.setattr(content, '_confirm_trial_navigation', lambda: False)
    before = (hub.pages.currentWidget(), hub.navigation.currentIndex(), window._adaptive_layout.page,
              window._center_stack.currentWidget(), window._acquisition_workspace_selected)
    for action in (window.acquisition_workspace_action, window.capture_preview_action,
                   window.diameter_experiment_action):
        action.trigger()
        after = (hub.pages.currentWidget(), hub.navigation.currentIndex(), window._adaptive_layout.page,
                 window._center_stack.currentWidget(), window._acquisition_workspace_selected)
        assert after == before
        assert window.acquisition_workspace_action.isChecked()
    assert window._activate_measure_workspace() is False
    assert hub.pages.currentWidget() is content
    # Cleanup must not be blocked by the synthetic guard.
    monkeypatch.setattr(content, '_confirm_trial_navigation', lambda: True)


def test_runtime_range_rules_clear_only_after_project_is_released(window, monkeypatch):
    from fdm.live_experiment import ExperimentSession
    from fdm.live_experiment.range_decisions import RangeRuleStore
    session = ExperimentSession()
    session._range_rules = rules = RangeRuleStore()
    rules._rules[(session.id, 'wool', 'above')] = {'action': 'keep'}
    window.project.live_experiments.append(session)
    monkeypatch.setattr(window, '_confirm_close_documents_for_transition', lambda *_: False)
    assert not window.close()
    assert rules.list_rules(session.id) == [{'action': 'keep'}]
    window._reset_workspace()
    assert not rules.list_rules(session.id)
    assert not window.project.live_experiments
