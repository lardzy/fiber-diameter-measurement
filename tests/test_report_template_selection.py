"""The report picker changes the actual live workbook, including same-path edits."""
from concurrent.futures import Future
import json
import threading

import pytest
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout

from fdm import settings
from fdm.live_experiment.excel_template import template_preferences, save_template_preferences
from fdm.ui import live_excel, trial_report_panel
from test_excel_live import FakeTransport, trial, wait_until


@pytest.fixture
def report_setup(tmp_path, monkeypatch, desktop_application):
    root = tmp_path / 'runtime' / 'content-templates'
    root.mkdir(parents=True)
    for name in ('sheet.xlt', 'sheet_dia.xlt'):
        (root / name).write_bytes(b'default template')
    monkeypatch.setattr(settings, 'runtime_directory', lambda: tmp_path / 'runtime')
    monkeypatch.setattr(settings, 'settings_directory', lambda: tmp_path / 'profile')
    transport = FakeTransport()
    manager = live_excel.LiveExcelManager(transport=transport, coalesce_ms=5)
    panels = []

    def make(mode='content'):
        window = QMainWindow()
        window.session = trial(mode)
        window.session.add_measurement(18.25, 'a')
        central = QWidget(window)
        window.setCentralWidget(central)
        window.report_layout = QVBoxLayout(central)
        window._guard = lambda fn: fn()
        panel = trial_report_panel.TrialReportPanel(window)
        panel.bind_live_excel(manager)
        # Keep the actual picker path: selecting a source must not spawn a
        # separate Excel inspection concurrently with workbook creation.
        panels.append((window, panel))
        return window, panel

    yield make, manager, transport, desktop_application
    for window, panel in panels:
        panel.close()
        window.close()
        window.deleteLater()
    transport.gate.set()
    manager.close()
    wait_until(desktop_application, lambda: manager.close())


@pytest.mark.parametrize('mode', ['content', 'diameter'])
def test_report_picker_replaces_unchanged_trial_and_same_path_again(report_setup, tmp_path, monkeypatch, mode):
    make, manager, transport, app = report_setup
    window, panel = make(mode)
    session = window.session
    manager.ensure(session)
    wait_until(app, lambda: manager.status(session.id)['state'] == 'synced')
    previous = session.to_dict()
    first_source = manager.status(session.id)['template_path']
    template = tmp_path / '用户选择的模板.xlt'
    template.write_bytes(b'custom first')
    monkeypatch.setattr(trial_report_panel.QFileDialog, 'getOpenFileName', lambda *a: (str(template), ''))
    def forbidden_inspect():
        pytest.fail('Selecting a template must not launch a parallel Excel inspector')
    monkeypatch.setattr(panel, 'inspect', forbidden_inspect)
    panel.choose()
    wait_until(app, lambda: manager.status(session.id)['generation'] == 2 and
               manager.status(session.id)['state'] == 'synced')
    assert manager.status(session.id)['template_path'] == str(template)
    assert template_preferences(mode)['template'] == str(template)
    assert str(template) in panel.live_template_label.text()
    assert first_source not in panel.live_template_label.text()
    template.write_bytes(b'custom second at identical path')
    panel.choose()
    wait_until(app, lambda: manager.status(session.id)['generation'] == 3 and
               manager.status(session.id)['state'] == 'synced')
    opens = [r for r in transport.requests if r[0] == 'open']
    assert [r[4]['template'] for r in opens] == [first_source, str(template), str(template)]
    assert opens[-1][4]['blocks'][0]['values'] == [[18.25]]
    assert not [r for r in transport.requests if r[0] == 'detach']
    assert panel.future is None
    assert session.to_dict() == previous
    assert template.read_bytes() == b'custom second at identical path'


@pytest.mark.parametrize('mode', ['content', 'diameter'])
def test_failed_same_path_picker_remains_visible_while_old_workbook_receives_new_measurements(report_setup, monkeypatch, mode):
    make, manager, transport, app = report_setup
    request = transport.request
    def request_with_identity(operation, *args, **kwargs):
        result = request(operation, *args, **kwargs)
        payload = args[3] if len(args) > 3 else kwargs.get('payload')
        if result.get('ok') and operation == 'open' and payload:
            result['template_identity'] = payload['template_identity']
        return result
    monkeypatch.setattr(transport, 'request', request_with_identity)
    window, panel = make(mode)
    session = window.session
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session))
    original = manager.template_binding(session)
    source = panel.path().resolve()
    source.write_bytes(b'new template contents at the same path')
    monkeypatch.setattr(trial_report_panel.QFileDialog, 'getOpenFileName', lambda *a: (str(source), ''))
    transport.failure = {'ok': False, 'error_code': 'template_mapping',
                         'message': '所选模板缺少原始数据页', 'workbook_connected': True}
    panel.choose()
    wait_until(app, lambda: manager.status(session.id)['state'] == 'error')
    session.add_measurement(21.5, 'a')
    manager.update(session)
    wait_until(app, lambda: manager.status(session.id)['state'] == 'synced' and manager.flush(session))
    assert manager.template_binding(session) == original
    text = panel.live_template_label.text()
    assert '更换未完成：' + str(source) in text
    assert '所选模板缺少原始数据页' in text
    assert '当前工作簿未替换' in text
    assert original['identity']['sha256'][:16] in text

    panel.choose()
    wait_until(app, lambda: manager.status(session.id)['generation'] == 2 and manager.flush(session))
    text = panel.live_template_label.text()
    assert '更换未完成' not in text
    assert '所选模板缺少原始数据页' not in text
    assert manager.template_binding(session)['identity']['sha256'][:16] in text


def test_selection_updates_new_trials_but_not_other_connected_books(report_setup, tmp_path, monkeypatch):
    make, manager, transport, app = report_setup
    first, panel = make()
    other, other_panel = make()
    for window in (first, other):
        manager.ensure(window.session)
        wait_until(app, lambda: manager.status(window.session.id)['state'] == 'synced')
    other_template = manager.status(other.session.id)['template_path']
    template = tmp_path / 'new-template.xlt'
    template.write_bytes(b'chosen')
    monkeypatch.setattr(trial_report_panel.QFileDialog, 'getOpenFileName', lambda *a: (str(template), ''))
    panel.choose()
    wait_until(app, lambda: manager.status(first.session.id)['generation'] == 2 and
               manager.status(first.session.id)['state'] == 'synced')
    # A stale hidden page must not overwrite the new choice when saving a method.
    other_panel.method_changed(0)
    assert template_preferences()['template'] == str(template)
    assert other_panel.path() == template
    other.session.add_measurement(21.5, 'a')
    manager.resync(other.session)
    wait_until(app, lambda: manager.status(other.session.id)['input_revision'] == other.session.revision and
               manager.status(other.session.id)['state'] == 'synced')
    assert manager.status(other.session.id)['generation'] == 1
    assert manager.status(other.session.id)['template_path'] == other_template
    assert len([r for r in transport.requests if r[0] == 'open' and r[1] == other.session.id]) == 1
    new, _ = make()
    manager.ensure(new.session)
    wait_until(app, lambda: manager.status(new.session.id)['state'] == 'synced')
    assert manager.status(new.session.id)['template_path'] == str(template)


def test_mapping_selection_uses_latest_template_and_rebinds(report_setup, tmp_path, monkeypatch):
    make, manager, transport, app = report_setup
    window, panel = make()
    template = tmp_path / 'latest.xlt'
    template.write_bytes(b'template')
    save_template_preferences({'template': str(template)})
    manager.ensure(window.session)
    wait_until(app, lambda: manager.status(window.session.id)['state'] == 'synced')
    mapping = tmp_path / 'mapping.json'
    mapping.write_text(json.dumps({'data_sheet': '我的原始数据'}), encoding='utf-8')
    monkeypatch.setattr(trial_report_panel.QFileDialog, 'getOpenFileName', lambda *a: (str(mapping), ''))
    panel.choose_mapping()
    wait_until(app, lambda: manager.status(window.session.id)['generation'] == 2 and
               manager.status(window.session.id)['state'] == 'synced')
    latest = [r for r in transport.requests if r[0] == 'open'][-1][4]
    assert latest['template'] == str(template)
    assert latest['profile']['data_sheet'] == '我的原始数据'


def test_missing_selection_or_running_report_task_keeps_preferences_and_book(report_setup, tmp_path, monkeypatch):
    make, manager, transport, app = report_setup
    window, panel = make()
    manager.ensure(window.session)
    wait_until(app, lambda: manager.status(window.session.id)['state'] == 'synced')
    monkeypatch.setattr(trial_report_panel.QFileDialog, 'getOpenFileName', lambda *a: (str(tmp_path / 'missing.xlt'), ''))
    with pytest.raises(FileNotFoundError):
        panel.choose()
    assert template_preferences() == {}
    assert len([r for r in transport.requests if r[0] == 'open']) == 1
    panel.future = Future()
    with pytest.raises(ValueError, match='已有模板任务'):
        panel.choose()
    assert template_preferences() == {}


@pytest.mark.parametrize('change', ['template', 'mapping', 'same_path_contents'])
def test_old_method_list_cannot_write_into_changed_template_preferences(report_setup, tmp_path, change):
    make, manager, transport, app = report_setup
    window, panel = make()
    template = panel.path(create=True)
    panel.method.addItem('旧模板专用标准', 'old-only')
    latest = {'template': str(template), 'method_labels': {window.session.method: 'new-template-standard'}}
    if change == 'template':
        replacement = tmp_path / 'other source.xlt'
        replacement.write_bytes(b'other source')
        latest['template'] = str(replacement)
    elif change == 'mapping':
        mapping = tmp_path / 'other mapping.json'
        mapping.write_text('{}', encoding='utf-8')
        latest['mapping'] = str(mapping)
    else:
        template.write_bytes(b'replaced source contents with a new standard list')
    save_template_preferences(latest)
    panel.method.setCurrentIndex(1)
    assert template_preferences() == latest
    assert panel.method.count() == 1
    assert '旧标准列表已清除' in panel.status.text()
    assert transport.requests == []


@pytest.mark.parametrize('change', ['template', 'mapping'])
def test_late_inspect_response_is_discarded_after_source_changes(report_setup, tmp_path, monkeypatch, change):
    make, manager, transport, app = report_setup
    window, panel = make()
    gate = threading.Event()

    def inspect_request(request, **kwargs):
        gate.wait(3)
        return {'methods': [{'label': 'stale inspected method'}]}

    monkeypatch.setattr(trial_report_panel, 'run_excel_request', inspect_request)
    try:
        trial_report_panel.TrialReportPanel.inspect(panel)
        preferences = template_preferences()
        replacement = tmp_path / ('new-source.xlt' if change == 'template' else 'new-mapping.json')
        replacement.write_bytes(b'fixture')
        preferences[change] = str(replacement)
        save_template_preferences(preferences)
        gate.set()
        wait_until(app, lambda: panel.future is None or panel.future.done())
        panel.poll()
        assert panel.method.count() == 1
        assert '已忽略旧模板返回' in panel.status.text()
        assert template_preferences() == preferences
    finally:
        gate.set()


def test_method_mapping_is_frozen_without_overwriting_templates_initial_standard(report_setup, tmp_path, monkeypatch):
    make, manager, transport, app = report_setup
    window, panel = make()
    session = window.session
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session))
    selected = tmp_path / 'custom methods.xlt'
    selected.write_bytes(b'template with custom method labels')
    monkeypatch.setattr(trial_report_panel.QFileDialog, 'getOpenFileName', lambda *a: (str(selected), ''))
    panel.choose()
    wait_until(app, lambda: manager.status(session.id)['generation'] == 2 and manager.flush(session))
    panel._task_source = panel._methods_source
    panel.future = Future()
    panel.future.set_result({'methods': [{'label': '自定义标准'}]})
    panel.poll()
    before = session.to_dict()
    panel.method.setCurrentIndex(1)
    # Simulate a different panel updating defaults while this open is queued.
    save_template_preferences({'template': str(selected), 'method_labels': {session.method: '其他标签'}})
    wait_until(app, lambda: manager.status(session.id)['generation'] == 3 and manager.flush(session))
    opened = [item for item in transport.requests if item[0] == 'open']
    assert all(item[4].get('report_settings') is None for item in opened)
    assert manager.template_binding(session)['preferences']['method_labels'][session.method] == '自定义标准'
    assert opened[-1][4]['template'] == str(selected)
    assert '旧工作簿保留' in panel.status.text()
    assert session.to_dict() == before
    assert not any(item[0] in ('close', 'save') for item in transport.requests)


def test_one_shot_export_uses_current_workbook_source_not_later_default(report_setup, tmp_path, monkeypatch):
    make, manager, transport, app = report_setup
    window, panel = make()
    manager.ensure(window.session)
    wait_until(app, lambda: manager.flush(window.session))
    bound = manager.status(window.session.id)['template_path']
    other = tmp_path / 'later-default.xlt'
    other.write_bytes(b'other workbook')
    save_template_preferences({'template': str(other)})
    captured = []
    monkeypatch.setattr(trial_report_panel.QFileDialog, 'getSaveFileName', lambda *a: (str(tmp_path/'export.xls'), ''))
    monkeypatch.setattr(trial_report_panel, 'run_excel_request', lambda request, **kwargs: captured.append(request) or {})
    panel.export()
    wait_until(app, lambda: bool(captured))
    assert captured[0]['template'] == bound
    assert captured[0]['template_identity']['sha256'] == manager.template_binding(window.session)['identity']['sha256']
    assert template_preferences()['template'] == str(other)


def test_report_page_can_repair_broken_config_and_diagnostic_is_bounded(report_setup, tmp_path, monkeypatch):
    from fdm.live_experiment.excel_template import template_preferences_path
    make, manager, transport, app = report_setup
    preferences = template_preferences_path()
    preferences.parent.mkdir(parents=True)
    preferences.write_text('{broken')
    window, panel = make()
    assert '无法读取试验模板设置' in panel.path_label.text()
    selected = tmp_path / 'repaired.xlt'
    selected.write_bytes(b'repaired')
    monkeypatch.setattr(trial_report_panel.QFileDialog, 'getOpenFileName', lambda *a: (str(selected), ''))
    panel.choose()
    assert template_preferences()['template'] == str(selected)
    panel.collect_diagnostics()
    first = json.loads(panel.diagnostics_text.toPlainText())
    panel.collect_diagnostics()
    second = json.loads(panel.diagnostics_text.toPlainText())
    assert second['previous_operation'] == first['previous_operation']
    assert second['selected_source']['sha256']
    assert second['session_id'] == window.session.id


def test_export_completion_uses_template_method_without_assuming_fdm_standard(report_setup, tmp_path):
    make, _, _, _ = report_setup
    window, panel = make()
    result = {'destination': str(tmp_path / 'report.xls'), 'method_source': 'template',
              'status': 'ready', 'warnings': [], 'comparisons': []}
    panel.future = Future()
    panel.future.set_result(result)
    panel.poll()
    assert panel.last_report == result
    assert '报告方法：由模板当前设置决定' in panel.status.text()
    assert '报告已生成' in window.statusBar().currentMessage()


def test_oneoff_template_task_keeps_global_lease_until_close_cleanup_finishes(report_setup):
    make, manager, transport, app = report_setup
    window, first = make()
    _, second = make()
    gate = threading.Event()
    first.start(lambda: gate.wait(3) or {'methods': []})
    try:
        with pytest.raises(ValueError, match='其他模板任务'):
            second.start(lambda: {})
        first.close()
        assert not manager.begin_template_inspection()
        gate.set()
        wait_until(app, lambda: first.future.done())
        # Completion callback queues release to the manager's owning Qt thread.
        wait_until(app, lambda: manager.begin_template_inspection())
        manager.end_template_inspection()
        second.start(lambda: {'methods': []})
        wait_until(app, lambda: second.future is None or second.future.done())
        second.poll()
        assert manager.begin_template_inspection()
        manager.end_template_inspection()
    finally:
        gate.set()
