"""Page preparation yields to navigation before device or workbook readiness."""
import time

import pytest
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from fdm.ui.main_window import MainWindow
from fdm.ui.live_experiment_workspace import LiveExperimentWorkspace


def until(predicate, timeout=3):
    deadline = time.perf_counter() + timeout
    while not predicate():
        QApplication.processEvents()
        if time.perf_counter() > deadline:
            raise AssertionError('page preparation did not reach expected state')
        QTest.qWait(1)


@pytest.fixture
def window(monkeypatch):
    monkeypatch.setattr('fdm.services.capture.available_capture_backends', lambda: [])
    monkeypatch.setattr('fdm.ui.trial_excel_lifecycle.ensure_live_excel', lambda host, session: None)
    host = MainWindow()
    host.resize(1280, 800)
    host.show()
    QApplication.processEvents()
    yield host
    monkeypatch.setattr(host, '_confirm_close_documents_for_transition', lambda *_: True)
    host.close()
    QApplication.processEvents()


def test_shell_is_shown_before_trial_stages_and_ready_page_is_reused(window, monkeypatch):
    starts = []
    monkeypatch.setattr(window, 'start_live_preview', lambda **kwargs: starts.append('start'))
    hub = window._open_acquisition_workbench(trial_index=1)
    assert hub.pages.currentWidget() is hub.loading_page
    assert hub._page_load_state == 'loading'
    assert hub._page_load['page'] is None
    assert not starts
    assert window._adaptive_layout.page.value == 'content_trial'
    assert not window._project_dock.isVisible()
    assert not window._inspector_dock.isVisible()
    until(lambda: hub._page_load_state in ('ready', 'error'))
    assert hub._page_load_state == 'ready', hub.loading_label.text()
    page = hub._current_trial()
    assert page is not None and page._initialized
    assert starts == ['start']
    names = [event['phase'] for event in hub.transition_timings()[-1]['events']]
    assert names.index('framework_painted') < names.index('trial_model_ready')
    assert names.index('trial_stage_1') < names.index('trial_stage_4') < names.index('page_interactive')
    hub.select_page(0)
    hub.select_page(1)
    assert hub._current_trial() is page
    assert hub._page_load is None


def test_slow_preparation_keeps_animation_and_navigation_responsive(window, monkeypatch):
    original = LiveExperimentWorkspace.initialize_next_stage
    gate = {'ready': False}
    monkeypatch.setattr(LiveExperimentWorkspace, 'initialize_next_stage',
                        lambda self: original(self) if gate['ready'] else False)
    hub = window._open_acquisition_workbench(trial_index=1, auto_start=False)
    until(lambda: hub._page_load is not None and hub._page_load['page'] is not None)
    assert not hub.loading_progress.isVisible()
    QTest.qWait(175)
    assert hub.loading_progress.isVisible()
    unfinished = hub._page_load['page']
    hub.select_page(0)
    assert hub.pages.currentWidget() is hub.capture_page
    assert unfinished._closed
    assert not hub.loading_progress.isVisible()
    gate['ready'] = True
    QTest.qWait(15)
    assert hub.pages.currentWidget() is hub.capture_page
    assert not getattr(window, '_live_experiment_workspaces', {})


def test_loading_leave_and_latest_page_request_do_not_publish_stale_page(window, monkeypatch):
    original = LiveExperimentWorkspace.initialize_next_stage
    gate = {'ready': False}
    monkeypatch.setattr(LiveExperimentWorkspace, 'initialize_next_stage',
                        lambda self: original(self) if gate['ready'] else False)
    hub = window._open_acquisition_workbench(trial_index=1, auto_start=False)
    until(lambda: hub._page_load is not None and hub._page_load['page'] is not None)
    first = hub._page_load['page']
    hub.select_page(2)
    until(lambda: hub._page_load is not None and hub._page_load['page'] is not None)
    second = hub._page_load['page']
    assert first._closed
    assert window._activate_measure_workspace()
    assert second._closed
    gate['ready'] = True
    QTest.qWait(15)
    assert window._center_stack.currentWidget() is not hub
    assert not window._acquisition_workspace_selected
    assert not getattr(window, '_live_experiment_workspaces', {})
    window._open_acquisition_workbench(trial_index=2, auto_start=False)
    until(lambda: hub._page_load_state == 'ready' and hub._current_trial() is not None)
    assert hub._current_trial().session.workspace_options['mode'] == 'diameter'


def test_failed_stage_can_retry_without_publishing_partial_trial(window, monkeypatch):
    original = LiveExperimentWorkspace._initialize_display
    fail = {'once': True}
    def initialize(self):
        if fail['once']:
            fail['once'] = False
            raise ValueError('simulated page setup failure')
        return original(self)
    monkeypatch.setattr(LiveExperimentWorkspace, '_initialize_display', initialize)
    hub = window._open_acquisition_workbench(trial_index=1, auto_start=False)
    until(lambda: hub._page_load_state == 'error')
    assert 'simulated page setup failure' in hub.loading_label.text()
    assert hub.loading_retry.isVisible()
    assert not getattr(window, '_live_experiment_workspaces', {})
    hub.loading_retry.click()
    until(lambda: hub._page_load_state in ('ready', 'error'))
    assert hub._page_load_state == 'ready', hub.loading_label.text()
    assert hub._current_trial()._initialized
    assert not hub.loading_progress.isVisible()


def test_project_retirement_during_loading_invalidates_queued_callbacks(window, monkeypatch):
    monkeypatch.setattr(LiveExperimentWorkspace, 'initialize_next_stage', lambda self: False)
    hub = window._open_acquisition_workbench(trial_index=1, auto_start=False)
    until(lambda: hub._page_load is not None and hub._page_load['page'] is not None)
    unfinished = hub._page_load['page']
    assert window._close_live_experiment()
    assert unfinished._closed
    assert hub.pages.currentWidget() is hub.capture_page
    assert hub._page_load_state == 'ready'
    QTest.qWait(10)
    assert not getattr(window, '_live_experiment_workspaces', {})


def test_leaving_before_first_paint_cancels_deferred_camera_start(window, monkeypatch):
    starts = []
    monkeypatch.setattr(window, 'start_live_preview', lambda **kwargs: starts.append('start'))
    hub = window._open_acquisition_workbench(trial_index=1)
    generation = hub.entry_generation
    assert window._activate_measure_workspace()
    QTest.qWait(25)
    assert not starts
    assert not hub.is_entry_current(generation)
    assert hub._page_load is None
    assert not getattr(window, '_live_experiment_workspaces', {})


def test_close_during_staged_construction_disposes_unpublished_controller(window, monkeypatch):
    monkeypatch.setattr(LiveExperimentWorkspace, 'initialize_next_stage', lambda self: False)
    hub = window._open_acquisition_workbench(trial_index=1, auto_start=False)
    until(lambda: hub._page_load is not None and hub._page_load['page'] is not None)
    page = hub._page_load['page']
    monkeypatch.setattr(window, '_confirm_close_documents_for_transition', lambda *_: True)
    assert window.close()
    QTest.qWait(15)
    assert page._closed and page.controller._closed
    assert hub._page_load is None
    assert not hub._entered


def test_excel_ready_records_current_inputs_once_per_visible_entry(window, monkeypatch):
    hub = window._open_acquisition_workbench(trial_index=1, auto_start=False)
    until(lambda: hub._page_load_state == 'ready')
    page = hub._current_trial()
    status = {'connected': True, 'active': True, 'input_pending': True,
              'unsynced': True, 'input_revision': page.session.revision - 1}
    monkeypatch.setattr(page.live_excel, 'status', lambda sid: status)
    phases = lambda: [event['phase'] for event in hub.transition_timings()[-1]['events']]
    page.live_excel.changed.emit(page.session.id)
    assert 'excel_ready' not in phases()
    # A stale acknowledgement is not readiness, even if a generic state says synced.
    status.update(input_pending=False, unsynced=False, state='synced')
    page.live_excel.changed.emit(page.session.id)
    assert 'excel_ready' not in phases()
    status['input_revision'] = page.session.revision
    page.live_excel.changed.emit('another-trial')
    assert 'excel_ready' not in phases()
    page.live_excel.changed.emit(page.session.id)
    page.live_excel.changed.emit(page.session.id)
    assert phases().count('excel_ready') == 1
    hub.select_page(0)
    page.live_excel.changed.emit(page.session.id)
    assert 'excel_ready' not in phases()
    hub.select_page(1)
    until(lambda: 'excel_ready' in phases())
    assert phases().count('excel_ready') == 1
    status.update(input_pending=True, unsynced=True)
    hub.select_page(0)
    hub.select_page(1)
    QTest.qWait(10)
    assert 'excel_ready' not in phases()
    assert window._activate_measure_workspace()
    status.update(input_pending=False, unsynced=False)
    page.live_excel.changed.emit(page.session.id)
    assert 'excel_ready' not in phases()


def test_capture_stage_feedback_clears_only_after_streaming(window):
    hub = window._open_acquisition_workbench(auto_start=False)
    manager = window._capture_manager
    manager._notify_capture_state('faulted', '模拟连接失败', 'open_device')
    assert '模拟连接失败' in hub.feedback.text()
    for stage, label in (('enumerate', '查找设备中'), ('open_device', '连接中'), ('first_frame', '等待第一帧')):
        manager._notify_capture_state('starting', stage=stage)
        assert label in hub.device_status.text()
        assert hub._error_message
    manager._notify_capture_state('streaming')
    assert not hub._error_message
    assert '采集中' in hub.device_status.text()
    assert '模拟连接失败' not in hub.feedback.text() or not hub.feedback.isVisible()
