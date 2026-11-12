"""Live workbook input diffs and serialized coordinator without Office."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import threading
import time

import pytest

from fdm.live_experiment import ExperimentSession
from fdm.live_experiment.models import MaterialSnapshot
from fdm.live_experiment.excel_template import (DiameterTemplateProfile, ExportSnapshot,
                                               TemplateProfile, template_request)
from fdm.live_experiment.excel_live import input_patch, project_inputs, PowerShellLiveTransport


def trial(mode='content'):
    session = ExperimentSession()
    session.workspace_options['mode'] = mode
    operator = session.add_operator('操作员甲')
    session.set_materials([MaterialSnapshot('a', '甲纤维', 1.3), MaterialSnapshot('b', '乙纤维', 1.5)],
                          operator_id=operator, direction='warp')
    return session


def projection(session):
    return project_inputs(ExportSnapshot.capture(session), 'template.xlt',
                          TemplateProfile.load(trial_mode=session.workspace_options['mode']))


def measurement_at(session, value, material, created_at=None):
    """Fixture chronology is independent of the host clock (including after deletion)."""
    if created_at is None:
        # Known identities include removed rows, so a later append never reuses
        # a deleted record's timestamp. Equal-time ordering has its own test.
        created_at = (datetime(2026, 11, 11, tzinfo=timezone.utc)
                      + timedelta(seconds=len(session._known_measurement_ids))).isoformat()
    record = session.add_measurement(value, material)
    record = replace(record, created_at=created_at)
    session.measurements[record.id] = record
    return record


@pytest.mark.parametrize('mode,column', [('content','D'), ('diameter','C')])
def test_append_is_single_column_delta_and_delete_clears_tail(mode, column):
    session = trial(mode)
    first = measurement_at(session, 12, 'a', '2026-11-11T00:00:00+00:00')
    old = projection(session)
    second = measurement_at(session, 18, 'a', '2026-11-11T00:00:01+00:00')
    patch = input_patch(old, projection(session))
    assert patch['blocks'] == [{'sheet': TemplateProfile.load(trial_mode=mode).data_sheet,
                               'cell': f'{column}26', 'values': [[18.0]]}]
    assert patch['clears'] == []
    assert patch['cells'] == []
    before = projection(session)
    session.delete_measurement(first.id)
    patch = input_patch(before, projection(session))
    assert patch['blocks'][0]['values'] == [[18.0]]
    assert patch['blocks'][0]['cell'] == f'{column}25'
    assert patch['clears'] == [{'sheet': TemplateProfile.load(trial_mode=mode).data_sheet,
                               'range': f'{column}26:{column}26'}]


def test_reclassification_rewrites_only_affected_columns():
    session = trial()
    identity = measurement_at(session, 12, 'a').id
    before = projection(session)
    record = session.measurements[identity]
    session.measurements[identity] = replace(record, material_id='b')
    session.revision += 1
    session._summary_cache = None
    patch = input_patch(before, projection(session))
    assert [block['cell'] for block in patch['blocks']] == ['E25']
    assert patch['clears'] == [{'sheet': '原始数据', 'range': 'D25:D25'}]


def test_evidence_only_revisions_do_not_write_workbook():
    session = trial()
    identity = measurement_at(session, 12, 'a').id
    before = projection(session)
    session.measurements[identity] = replace(session.measurements[identity], evidence={'saved': True})
    session.revision += 1
    assert input_patch(before, projection(session)) is None


def test_empty_metadata_and_full_reset_cover_absent_participants():
    session = trial('diameter')
    value = projection(session)
    patch = input_patch(None, value)
    assert {'sheet': '直径实验原始数据', 'range': 'K25:K65536'} in patch['clears']
    assert {'sheet': '直径实验原始数据', 'range': 'K6'} not in patch['clears']
    assert 'managed_ranges' not in patch
    assert not any(item.get('sheet') == '新报告-数据' for item in patch['cells'])


def test_diameter_input_aggregates_operator_records_without_content_formula():
    session = trial('diameter')
    measurement_at(session, 12.123456789, 'a')
    measurement_at(session, 16, 'b')
    request = template_request(ExportSnapshot.capture(session), 'sheet_dia.xlt', 'report.xls')
    assert request['trial_kind'] == 'diameter'
    assert request['blocks'] == [{'cell':'C25', 'values':[[12.123456789], [16.0]]}]
    assert request['checks'][0]['expected'] == 2
    assert 'content_cell' not in request['checks'][0]
    assert request['profile']['statistical_capacity'] == 2976
    assert request['profile']['group_width_cell'] == 'F17'


def test_projection_is_detached_from_session_and_serializable():
    session = trial()
    measurement_at(session, 12, 'a')
    before = projection(session)
    encoded = before.encoded
    returned = before.data()
    returned['raw']['D'].append(999)
    measurement_at(session, 18, 'a')
    assert before.encoded == encoded
    assert before.data()['raw']['D'] == [12]
    assert 'evidence' not in encoded


class FakeTransport:
    def __init__(self):
        self.requests = []
        self.gate = threading.Event()
        self.gate.set()
        self.closed = False
        self.failure = None

    def request(self, operation, session_id='', generation=0, input_revision=0, payload=None, *, progress=None):
        self.requests.append((operation, session_id, generation, input_revision, payload))
        self.gate.wait(3)
        if self.failure:
            failure, self.failure = self.failure, None
            return failure
        if progress and operation in ('open','sync'):
            progress({'event':'inputs_written','session_id':session_id,'generation':generation,
                      'input_revision':input_revision, 'elapsed_ms': 3,
                      'batch_id':payload['batch_id'], 'input_written':True})
        return {'ok':True, 'status':'synced', 'saved':False, 'save_state':'never_saved',
                'input_written':operation in ('open','sync'), 'name':'测试工作簿'}

    def close(self):
        self.closed = True


def wait_until(app, predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while not predicate():
        app.processEvents()
        if time.monotonic() >= deadline:
            raise AssertionError('condition not reached')
        time.sleep(.003)
    app.processEvents()


@pytest.fixture
def coordinator(tmp_path, monkeypatch, desktop_application):
    from fdm.ui import live_excel
    template = tmp_path/'中文模板.xlt'
    template.write_bytes(b'fixture')
    monkeypatch.setattr(live_excel, 'working_template', lambda **kwargs: template)
    monkeypatch.setattr(live_excel, 'template_preferences', lambda **kwargs: {})
    transport = FakeTransport()
    manager = live_excel.LiveExcelManager(transport=transport, coalesce_ms=5)
    yield manager, transport, desktop_application
    transport.gate.set()
    manager.close()
    wait_until(desktop_application, lambda: manager.close())


def test_manager_does_not_start_for_update_on_hidden_untracked_session(coordinator):
    manager, transport, app = coordinator
    session = trial()
    manager.update(session)
    manager._pump()
    assert transport.requests == []
    assert manager.status(session.id)['state'] == 'idle'


def test_failed_inflight_template_does_not_block_latest_queued_selection(coordinator, tmp_path):
    manager, transport, app = coordinator
    session = trial()
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session))
    old_binding = manager.template_binding(session)
    first, latest = tmp_path/'invalid-a.xlt', tmp_path/'valid-b.xlt'
    first.write_bytes(b'invalid mapping')
    latest.write_bytes(b'latest template')
    transport.gate.clear()
    transport.failure = {'ok': False, 'error_code': 'template_mapping', 'message': 'A mapping invalid',
                         'workbook_connected': True}
    manager.replace_template(session, {'template': str(first)})
    wait_until(app, lambda: len([r for r in transport.requests if r[0] == 'open']) == 2)
    manager.replace_template(session, {'template': str(latest)})
    assert manager.template_binding(session) == old_binding
    transport.gate.set()
    wait_until(app, lambda: manager.template_binding(session)['path'] == str(latest) and manager.flush(session))
    opens = [r for r in transport.requests if r[0] == 'open']
    assert [r[4]['template'] for r in opens][-2:] == [str(first), str(latest)]
    assert manager.status(session.id)['generation'] == 2
    assert not manager._links[session.id].failed_template
    assert not manager.status(session.id)['pending_template_path']
    assert not any(r[0] == 'detach' for r in transport.requests)


def test_reconnect_after_transport_failure_uses_latest_selection_not_failed_earlier_one(coordinator, tmp_path, monkeypatch):
    from fdm.ui import live_excel
    manager, transport, app = coordinator
    session = trial()
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session))
    first, latest = tmp_path/'failed-a.xlt', tmp_path/'latest-b.xlt'
    first.write_bytes(b'first')
    latest.write_bytes(b'latest')
    recovered = FakeTransport()
    monkeypatch.setattr(live_excel, 'PowerShellLiveTransport', lambda: recovered)
    transport.gate.clear()
    transport.failure = {'ok': False, 'error_code': 'worker_disconnected', 'message': 'lost transport'}
    manager.replace_template(session, {'template': str(first)})
    wait_until(app, lambda: len([r for r in transport.requests if r[0] == 'open']) == 2)
    manager.replace_template(session, {'template': str(latest)})
    transport.gate.set()
    wait_until(app, lambda: manager.status(session.id)['state'] == 'error')
    assert manager.status(session.id)['pending_template_path'] == str(latest)
    manager.resync(session)
    wait_until(app, lambda: manager.template_binding(session)['path'] == str(latest) and manager.flush(session))
    assert [r[4]['template'] for r in recovered.requests if r[0] == 'open'] == [str(latest)]


@pytest.mark.parametrize('resolution', ['recreate', 'resync', 'detach'])
def test_same_path_replacement_failure_survives_old_workbook_sync_until_explicit_resolution(coordinator, resolution):
    from fdm.ui.live_excel import LiveExcelControls
    manager, transport, app = coordinator
    session = trial()
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session))
    original = manager.template_binding(session)
    source = Path(original['path'])
    source.write_bytes(b'replaced source has invalid mapping')
    transport.failure = {'ok': False, 'error_code': 'template_mapping',
                         'message': 'new template is missing its data sheet', 'workbook_connected': True}
    manager.replace_template(session, {'template': str(source)})
    wait_until(app, lambda: manager.status(session.id)['state'] == 'error')
    failure = manager.status(session.id)['template_replacement_message']
    assert 'new template is missing its data sheet' in failure

    measurement_at(session, 12, 'a')
    manager.update(session)
    wait_until(app, lambda: manager.status(session.id)['state'] == 'synced' and manager.flush(session))
    manager.show(session)
    wait_until(app, lambda: not manager._busy and not manager._links[session.id].actions)
    state = manager.status(session.id)
    assert state['input_revision'] == session.revision
    assert manager.template_binding(session) == original
    assert state['generation'] == 1
    assert state['template_replacement_failed'] is True
    assert state['template_replacement_path'] == str(source)
    assert state['pending_template_path'] == str(source)
    assert state['template_replacement_message'] == failure
    assert state['template_replacement_error_code'] == 'template_mapping'
    assert 'raw_error' not in state  # Normal input/status errors are independent.
    assert 'new template is missing its data sheet' in LiveExcelControls._diagnostics(state)

    if resolution in ('recreate', 'resync'):
        transport.gate.clear()
        if resolution == 'recreate':
            manager.replace_template(session, {'template': str(source)})
        else:
            manager.resync(session)
        assert not manager.status(session.id)['template_replacement_failed']
        assert not manager.status(session.id)['template_replacement_message']
        transport.gate.set()
        wait_until(app, lambda: manager.status(session.id)['generation'] == 2 and manager.flush(session))
        assert manager.template_binding(session)['identity']['sha256'] != original['identity']['sha256']
    else:
        manager.stop(session)
        wait_until(app, lambda: manager.is_detached(session))
    final = manager.status(session.id)
    assert not final['template_replacement_failed']
    assert not final['template_replacement_path']
    assert not final['template_replacement_message']
    assert not final['pending_template_path']


def test_disconnected_local_detach_clears_replacement_failure(coordinator):
    manager, transport, app = coordinator
    session = trial()
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session))
    source = manager.template_binding(session)['path']
    transport.failure = {'ok': False, 'error_code': 'workbook_closed',
                         'message': 'workbook closed during replacement', 'workbook_connected': False}
    manager.replace_template(session, {'template': source})
    wait_until(app, lambda: manager.status(session.id)['template_replacement_failed'])
    assert not manager.status(session.id)['connected']
    manager.stop(session)
    wait_until(app, lambda: manager.is_detached(session))
    state = manager.status(session.id)
    assert not any(request[0] == 'detach' for request in transport.requests)
    assert not state['template_replacement_failed']
    assert not state['template_replacement_path']
    assert not state['template_replacement_message']
    assert not state['pending_template_path']
    assert manager._links[session.id].failed_template is None


def test_manager_keeps_one_flight_latest_snapshot_and_reuses_workbook(coordinator):
    manager, transport, app = coordinator
    session = trial()
    transport.gate.clear()
    manager.ensure(session)
    wait_until(app, lambda: len(transport.requests) == 1)
    for diameter in (12, 14, 16):
        measurement_at(session, diameter, 'a')
        manager.update(session)
    assert len(transport.requests) == 1
    transport.gate.set()
    wait_until(app, lambda: manager.flush(session))
    mutations = [request for request in transport.requests if request[0] in ('open','sync')]
    assert [item[0] for item in mutations] == ['open', 'sync']
    assert mutations[-1][4]['blocks'][0]['values'] == [[12], [14], [16]]
    manager.ensure(session)
    app.processEvents()
    assert len([r for r in transport.requests if r[0] == 'open']) == 1


def test_evidence_update_acknowledges_revision_without_com_request(coordinator):
    manager, transport, app = coordinator
    session = trial()
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session))
    before = len([r for r in transport.requests if r[0] in ('open','sync')])
    poll_at = manager._links[session.id].next_poll
    session.revision += 1
    manager.update(session)
    wait_until(app, lambda: manager.flush(session))
    assert len([r for r in transport.requests if r[0] in ('open','sync')]) == before
    assert manager._links[session.id].next_poll == poll_at


def test_stop_does_not_close_or_save_workbook_and_reentry_is_inert(coordinator):
    manager, transport, app = coordinator
    session = trial()
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session))
    manager.stop(session.id)
    wait_until(app, lambda: manager.is_detached(session.id))
    manager.ensure(session)
    manager._pump()
    assert transport.requests[-1][0] == 'detach'
    assert manager.status(session.id)['active'] is False
    assert 'save' not in [r[0] for r in transport.requests]


def test_failed_write_never_marks_revision_synced(coordinator):
    manager, transport, app = coordinator
    session = trial()
    transport.failure = {'ok':False,'error_code':'excel_failed','message':'宏失败','needs_full_sync':True}
    manager.ensure(session)
    wait_until(app, lambda: manager.status(session.id)['state'] == 'error')
    assert manager.status(session.id)['state'] == 'error'
    assert manager.flush(session)  # No workbook was ever established.
    manager.resync(session)
    wait_until(app, lambda: manager.flush(session))
    assert [r for r in transport.requests if r[0] in ('open','sync')][-1][4]['full_sync']


def test_manual_workbook_close_requires_explicit_resync(coordinator):
    manager, transport, app = coordinator
    session = trial()
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session))
    transport.failure = {'ok':False,'error_code':'workbook_closed','message':'工作簿已关闭'}
    manager.show(session.id)
    wait_until(app, lambda: manager.status(session.id)['state'] == 'stopped')
    manager.ensure(session)
    manager._pump()
    assert len([r for r in transport.requests if r[0] == 'open']) == 1
    manager.resync(session)
    wait_until(app, lambda: manager.flush(session))
    assert len([r for r in transport.requests if r[0] == 'open']) == 2


def test_template_kind_cannot_silently_write_other_trials_mapping():
    with pytest.raises(ValueError, match='类型'):
        template_request(ExportSnapshot.capture(trial('diameter')), 'x.xlt', 'x.xls', TemplateProfile())


def test_resync_during_inflight_preserves_latest_full_rebuild(coordinator):
    manager, transport, app = coordinator
    session = trial()
    transport.gate.clear()
    manager.ensure(session)
    wait_until(app, lambda: len(transport.requests) == 1)
    measurement_at(session, 23, 'a')
    manager.resync(session)
    transport.gate.set()
    wait_until(app, lambda: manager.flush(session))
    requests = [item for item in transport.requests if item[0] in ('open','sync')]
    assert len(requests) == 2
    assert requests[-1][4]['full_sync']
    assert requests[-1][4]['blocks'][0]['values'] == [[23]]


def test_explicit_template_selection_reassociates_with_new_generation(coordinator, tmp_path):
    manager, transport, app = coordinator
    session = trial()
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session))
    old_generation = manager.status(session.id)['generation']
    replacement = tmp_path/'other.xlt'
    replacement.write_bytes(b'changed fixture')
    manager.replace_template(session, {'template': str(replacement)})
    wait_until(app, lambda: manager.flush(session))
    requests = [item for item in transport.requests if item[0] == 'open']
    assert len(requests) == 2
    assert requests[-1][2] == old_generation + 1
    assert requests[-1][4]['template'] == str(replacement)
    assert not any(item[0] == 'detach' for item in transport.requests)
    assert manager.status(session.id)['template_path'] == str(replacement)
    assert not manager.status(session.id)['template_replacement_pending']


def test_same_path_reselection_reads_template_again_without_revision_change(coordinator):
    manager, transport, app = coordinator
    session = trial()
    measurement_at(session, 18, 'a')
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session))
    template = Path(manager.status(session.id)['template_path'])
    template.write_bytes(b'replaced at the same filename')
    revision = session.revision
    manager.replace_template(session, {'template': str(template)})
    wait_until(app, lambda: manager.flush(session))
    opens = [item for item in transport.requests if item[0] == 'open']
    assert len(opens) == 2
    assert opens[-1][2] == opens[0][2] + 1
    assert opens[-1][3] == revision
    assert opens[-1][4]['full_sync']
    assert opens[-1][4]['blocks'][0]['values'] == [[18]]
    assert not any(item[0] in ('save', 'close') for item in transport.requests)


def test_existing_links_keep_bound_template_and_mapping_when_defaults_change(coordinator, tmp_path, monkeypatch):
    from fdm.ui import live_excel
    manager, transport, app = coordinator
    first, second = trial(), trial()
    for session in (first, second):
        manager.ensure(session)
        wait_until(app, lambda: manager.flush(session))
    old_path = manager.status(first.id)['template_path']
    replacement = tmp_path / 'next trial.xlt'
    replacement.write_bytes(b'new default')
    monkeypatch.setattr(live_excel, 'working_template', lambda **kwargs: replacement)
    monkeypatch.setattr(live_excel, 'template_preferences', lambda **kwargs: {'mapping': 'missing-new-map.json'})
    Path(old_path).unlink()  # Existing workbooks do not need their source file.
    measurement_at(first, 12, 'a')
    manager.update(first)
    manager.resync(second)
    wait_until(app, lambda: manager.flush(first) and manager.flush(second))
    assert len([item for item in transport.requests if item[0] == 'open']) == 2
    assert manager.status(first.id)['template_path'] == old_path
    assert manager.status(second.id)['template_path'] == old_path
    assert all(item[4]['template'] == old_path for item in transport.requests if item[0] == 'sync')
    monkeypatch.setattr(live_excel, 'template_preferences', lambda **kwargs: {})
    third = trial()
    manager.ensure(third)
    wait_until(app, lambda: manager.flush(third))
    assert manager.status(third.id)['template_path'] == str(replacement)


def test_missing_replacement_preserves_old_generation_and_detaches_safely(coordinator, tmp_path):
    manager, transport, app = coordinator
    session = trial()
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session))
    old = manager.status(session.id)
    before = len(transport.requests)
    missing = tmp_path / 'missing selected template.xlt'
    manager.replace_template(session, {'template': str(missing)})
    wait_until(app, lambda: manager.status(session.id)['state'] == 'error')
    state = manager.status(session.id)
    assert state['generation'] == old['generation']
    assert state['template_path'] == old['template_path']
    assert state['pending_template_path'] == str(missing)
    assert state['connected']
    assert '未使用其他默认模板' in state['message']
    assert len(transport.requests) == before
    manager.stop(session.id)
    wait_until(app, lambda: manager.is_detached(session.id))
    assert transport.requests[-1][0] == 'detach'
    assert transport.requests[-1][2] == old['generation']


def test_reconnect_after_manual_close_retains_this_trials_template(coordinator, tmp_path, monkeypatch):
    from fdm.ui import live_excel
    manager, transport, app = coordinator
    session = trial()
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session))
    original = manager.status(session.id)['template_path']
    other = tmp_path / 'other trials default.xlt'
    other.write_bytes(b'other')
    monkeypatch.setattr(live_excel, 'working_template', lambda **kwargs: other)
    transport.failure = {'ok': False, 'error_code': 'workbook_closed'}
    manager.show(session.id)
    wait_until(app, lambda: manager.status(session.id)['state'] == 'stopped')
    manager.resync(session)
    wait_until(app, lambda: manager.flush(session))
    opens = [item for item in transport.requests if item[0] == 'open']
    assert len(opens) == 2
    assert opens[-1][4]['template'] == original


def test_failed_new_workbook_retry_keeps_selected_template(coordinator, tmp_path):
    manager, transport, app = coordinator
    session = trial()
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session))
    old_generation = manager.status(session.id)['generation']
    request = transport.request
    failed = False

    def failed_open(operation, session_id='', generation=0, input_revision=0, payload=None, *, progress=None):
        nonlocal failed
        if operation == 'open' and generation > old_generation and not failed:
            failed = True
            transport.requests.append((operation, session_id, generation, input_revision, payload))
            return {'ok': False, 'error_code': 'template_mapping', 'workbook_connected': False}
        return request(operation, session_id, generation, input_revision, payload, progress=progress)

    transport.request = failed_open
    selected = tmp_path / 'explicitly selected.xlt'
    selected.write_bytes(b'new')
    manager.replace_template(session, {'template': str(selected)})
    wait_until(app, lambda: manager.status(session.id)['state'] == 'error')
    assert manager.status(session.id)['connected']
    assert manager.status(session.id)['generation'] == old_generation
    assert manager.status(session.id)['template_path'] != str(selected)
    manager.resync(session)
    wait_until(app, lambda: manager.flush(session))
    opens = [item for item in transport.requests if item[0] == 'open']
    assert len(opens) == 3
    assert opens[-1][4]['template'] == str(selected)
    assert manager.status(session.id)['pending_template_path'] == ''


def test_template_candidate_busy_keeps_old_generation_until_retry(coordinator, tmp_path):
    manager, transport, app = coordinator
    session = trial()
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session))
    old_generation = manager.status(session.id)['generation']
    selected = tmp_path / 'selected after detach retry.xlt'
    selected.write_bytes(b'new')
    transport.failure = {'ok': False, 'error_code': 'excel_busy', 'workbook_connected': True}
    manager.replace_template(session, {'template': str(selected)})
    wait_until(app, lambda: manager.status(session.id)['state'] == 'busy')
    assert manager.status(session.id)['generation'] == old_generation
    assert manager.status(session.id)['template_replacement_pending']
    manager._links[session.id].retry_at = 0
    wait_until(app, lambda: manager.flush(session))
    assert manager.status(session.id)['generation'] == old_generation + 1
    assert manager.status(session.id)['template_path'] == str(selected)
    assert not any(item[0] == 'detach' for item in transport.requests)
    attempts = [item for item in transport.requests if item[0] == 'open' and item[2] > old_generation]
    assert len(attempts) == 2
    assert attempts[0][4] == attempts[1][4]


def test_busy_old_batch_finishes_before_latest_selected_template(coordinator, tmp_path):
    manager, transport, app = coordinator
    session = trial()
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session))
    measurement_at(session, 12, 'a')
    transport.failure = {'ok': False, 'error_code': 'excel_busy', 'workbook_connected': True}
    manager.update(session)
    wait_until(app, lambda: manager.status(session.id)['state'] == 'busy')
    pending = manager._links[session.id].pending_batch
    transport.gate.clear()
    earlier, latest = tmp_path / 'earlier.xlt', tmp_path / 'latest.xlt'
    earlier.write_bytes(b'earlier')
    latest.write_bytes(b'latest')
    manager.replace_template(session, {'template': str(earlier)})
    wait_until(app, lambda: manager._busy is not None)
    prefs = {'template': str(latest)}
    manager.replace_template(session, prefs)
    prefs['template'] = str(earlier)  # The selected configuration is detached.
    measurement_at(session, 24, 'a')
    manager.update(session)
    assert not manager.flush(session)
    assert manager.status(session.id)['pending_template_path'] == str(latest)
    transport.gate.set()
    wait_until(app, lambda: manager.flush(session))
    mutations = [item for item in transport.requests if item[0] in ('open', 'sync', 'detach')]
    assert [item[0] for item in mutations] == ['open', 'sync', 'sync', 'open']
    assert mutations[1][4]['batch_id'] == mutations[2][4]['batch_id'] == pending.payload['batch_id']
    assert mutations[-1][4]['template'] == str(latest)
    assert mutations[-1][4]['blocks'][0]['values'] == [[12], [24]]


@pytest.mark.parametrize('shutdown', [False, True])
def test_stop_or_close_cancels_queued_template_selection(coordinator, tmp_path, shutdown):
    manager, transport, app = coordinator
    session = trial()
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session))
    transport.gate.clear()
    measurement_at(session, 12, 'a')
    manager.update(session)
    wait_until(app, lambda: manager._busy is not None)
    template = tmp_path / 'never opened.xlt'
    template.write_bytes(b'new')
    manager.replace_template(session, {'template': str(template)})
    if shutdown:
        manager.close()
    else:
        manager.stop(session.id)
    transport.gate.set()
    wait_until(app, lambda: manager.close() if shutdown else manager.is_detached(session.id))
    assert len([item for item in transport.requests if item[0] == 'open']) == 1
    assert not manager.status(session.id)['template_replacement_pending']


@pytest.mark.parametrize('new_workbook_connected', [False, True])
def test_stop_during_new_template_busy_uses_actual_association_generation(coordinator, tmp_path, new_workbook_connected):
    manager, transport, app = coordinator
    session = trial()
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session))
    old_generation = manager.status(session.id)['generation']
    request = transport.request

    def busy_open(operation, session_id='', generation=0, input_revision=0, payload=None, *, progress=None):
        if operation == 'open' and generation > old_generation:
            transport.requests.append((operation, session_id, generation, input_revision, payload))
            return {'ok': False, 'error_code': 'excel_busy', 'workbook_connected': new_workbook_connected}
        return request(operation, session_id, generation, input_revision, payload, progress=progress)

    transport.request = busy_open
    template = tmp_path / 'selected.xlt'
    template.write_bytes(b'new')
    manager.replace_template(session, {'template': str(template)})
    wait_until(app, lambda: manager.status(session.id)['state'] == 'busy')
    assert manager.status(session.id)['connected']
    assert manager.status(session.id)['generation'] == old_generation
    manager.stop(session.id)
    wait_until(app, lambda: manager.is_detached(session.id))
    detaches = [item for item in transport.requests if item[0] == 'detach']
    assert [item[2] for item in detaches] == [old_generation]
    assert not manager.status(session.id)['template_replacement_pending']


def test_open_preserves_template_settings_and_records_source_identity(coordinator):
    manager, transport, app = coordinator
    session = trial()
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session))
    payload = next(item[4] for item in transport.requests if item[0] == 'open')
    assert payload['report_settings'] is None
    assert payload['report_cells']  # Initial sample/operator information is separate.
    identity = payload['template_identity']
    assert len(identity['sha256']) == 64
    assert identity['size'] == len(b'fixture')
    assert manager.template_binding(session)['identity'] == identity


def test_inspection_lease_blocks_new_excel_creation_but_keeps_existing_sync(coordinator):
    manager, transport, app = coordinator
    first, second = trial(), trial()
    manager.ensure(first)
    wait_until(app, lambda: manager.flush(first))
    assert manager.begin_template_inspection()
    assert not manager.begin_template_inspection()
    manager.ensure(second)
    manager._pump()
    assert len([item for item in transport.requests if item[0] == 'open']) == 1
    measurement_at(first, 23, 'a')
    manager.update(first)
    wait_until(app, lambda: manager.flush(first))
    assert len([item for item in transport.requests if item[0] == 'open']) == 1
    manager.end_template_inspection()
    wait_until(app, lambda: manager.flush(second))
    assert len([item for item in transport.requests if item[0] == 'open']) == 2


def test_inspection_cannot_acquire_lease_during_pending_workbook_creation(coordinator):
    manager, transport, app = coordinator
    transport.gate.clear()
    manager.ensure(trial())
    wait_until(app, lambda: len(transport.requests) == 1)
    assert not manager.begin_template_inspection()
    transport.gate.set()


def test_reentry_detects_changed_source_without_recreating_existing_book(coordinator):
    manager, transport, app = coordinator
    session = trial()
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session))
    original = manager.template_binding(session)
    Path(original['path']).write_bytes(b'another template at the same filename')
    manager.ensure(session)
    wait_until(app, lambda: manager.status(session.id).get('template_source_changed'))
    assert manager.template_binding(session) == original
    assert len([item for item in transport.requests if item[0] == 'open']) == 1
    assert manager.status(session.id)['connected']


def test_missing_source_on_reentry_does_not_stop_existing_workbook(coordinator):
    manager, transport, app = coordinator
    session = trial()
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session))
    Path(manager.template_binding(session)['path']).unlink()
    manager.ensure(session)
    wait_until(app, lambda: manager.status(session.id).get('template_source_error'))
    measurement_at(session, 19, 'a')
    manager.update(session)
    wait_until(app, lambda: manager.flush(session))
    assert manager.status(session.id)['connected']
    assert [item for item in transport.requests if item[0] == 'sync'][-1][4]['blocks'][0]['values'] == [[19]]


@pytest.mark.parametrize('missing_at_open', [False, True])
def test_unavailable_diagnostic_hash_does_not_block_open_or_existing_sync(coordinator, monkeypatch, missing_at_open):
    from fdm.ui import live_excel
    manager, transport, app = coordinator
    session = trial()
    real_identity = live_excel.template_identity

    def without_hash(path):
        identity = real_identity(path)
        return {**identity, 'sha256': None, 'hash_status': 'unavailable', 'hash_error': 'diagnostic only'}

    if missing_at_open:
        monkeypatch.setattr(live_excel, 'template_identity', without_hash)
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session))
    original = manager.template_binding(session)
    monkeypatch.setattr(live_excel, 'template_identity', without_hash)
    manager.ensure(session)
    wait_until(app, lambda: 'template_source_changed' in manager.status(session.id))
    assert not manager.status(session.id)['template_source_changed']
    assert not manager.status(session.id).get('template_source_error')
    assert manager.template_binding(session) == original
    measurement_at(session, 19, 'a')
    manager.update(session)
    wait_until(app, lambda: manager.flush(session))
    assert manager.status(session.id)['connected']
    assert len([item for item in transport.requests if item[0] == 'open']) == 1
    assert [item for item in transport.requests if item[0] == 'sync'][-1][4]['blocks'][0]['values'] == [[19]]


def test_failed_replacement_keeps_exact_input_baseline_and_binding(coordinator, tmp_path):
    manager, transport, app = coordinator
    session = trial()
    measurement_at(session, 17, 'a')
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session))
    old_binding = manager.template_binding(session)
    old_projection = manager._links[session.id].projection
    selected = tmp_path / 'bad mapping.xlt'
    selected.write_bytes(b'candidate')
    transport.failure = {'ok': False, 'error_code': 'template_mapping', 'candidate_pending': False}
    manager.replace_template(session, {'template': str(selected)})
    wait_until(app, lambda: manager.status(session.id)['state'] == 'error')
    assert manager.template_binding(session) == old_binding
    assert manager._links[session.id].projection is old_projection
    assert not any(item[0] == 'detach' for item in transport.requests)
    measurement_at(session, 18, 'a')
    manager.update(session)
    wait_until(app, lambda: manager._links[session.id].acknowledged_revision == session.revision)
    last = [item for item in transport.requests if item[0] == 'sync'][-1][4]
    assert last['template'] == old_binding['path']
    assert last['blocks'] == [{'sheet': '原始数据', 'cell': 'D26', 'values': [[18]]}]


def test_never_established_workbook_does_not_block_exit(coordinator):
    manager, transport, app = coordinator
    session = trial()
    transport.failure = {'ok':False, 'error_code':'unsupported_platform', 'message':'需要 Windows Excel'}
    manager.ensure(session)
    wait_until(app, lambda: manager.status(session.id)['state'] == 'unsupported')
    assert manager.ready_to_detach(session)
    assert not manager.status(session.id)['unsynced']
    assert not manager.status(session.id)['pending']
    assert manager.detach(session)


def test_dirty_existing_workbook_is_not_reported_flushed_after_failed_calculation(coordinator):
    manager, transport, app = coordinator
    session = trial()
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session))
    measurement_at(session, 14, 'a')
    transport.failure = {'ok':False,'error_code':'excel_failed','message':'宏失败','needs_full_sync':True}
    manager.update(session)
    wait_until(app, lambda: manager.status(session.id)['state'] == 'error')
    assert not manager.flush(session)
    assert manager.status(session.id)['unsynced']
    assert manager.status(session.id)['ever_connected']
    assert not manager.detach(session)
    manager.detach(session, force=True)
    wait_until(app, lambda: manager.is_detached(session.id))


_real_transport_request = PowerShellLiveTransport.request


def transport_fixture(monkeypatch, responses):
    """Exercise the actual framing/correlation loop with no process or Office."""
    from types import SimpleNamespace
    transport = PowerShellLiveTransport(timeout=.02)
    class Input:
        def write(self, line):
            request = json.loads(line)
            for content in responses:
                reply = {key: request[key] for key in ('request_id','session_id','generation','input_revision')}
                reply.update(content)
                transport._lines.put(json.dumps(reply))
        def flush(self): pass
    process = SimpleNamespace(stdin=Input(), poll=lambda: None, terminated=False)
    process.terminate = lambda: setattr(process, 'terminated', True)
    transport._process = process
    monkeypatch.setattr(transport, '_start', lambda: None)
    monkeypatch.setattr(transport, 'request', _real_transport_request.__get__(transport))
    return transport, process


def test_transport_progress_is_not_final_ack(monkeypatch):
    transport, process = transport_fixture(monkeypatch, [
        {'event':'inputs_written','status':'calculating'}, {'ok':True,'status':'synced'}])
    progress = []
    result = transport.request('sync', 'trial-a', 5, 8, progress=progress.append)
    assert result['ok']
    assert [item['event'] for item in progress] == ['inputs_written']
    assert not process.terminated


@pytest.mark.parametrize('wrong', [{'request_id':'old'}, {'session_id':'wrong'}, {'generation':0}, {'input_revision':1}])
def test_transport_rejects_late_or_cross_session_ack(monkeypatch, wrong):
    from fdm.live_experiment.excel_template import ExcelTemplateError
    transport, process = transport_fixture(monkeypatch, [{'ok':True, **wrong}])
    with pytest.raises(ExcelTemplateError) as failure:
        transport.request('sync','trial-a',5,8)
    assert failure.value.code == 'protocol_mismatch'
    assert process.terminated  # only the owned PS process, no Excel PID action


def test_transport_timeout_invalidates_stream_and_does_not_resend(monkeypatch):
    from fdm.live_experiment.excel_template import ExcelTemplateError
    transport, process = transport_fixture(monkeypatch, [])
    with pytest.raises(ExcelTemplateError) as failure:
        transport.request('sync','trial-a',5,8)
    assert failure.value.code == 'worker_timeout'
    assert transport._closed
    assert process.terminated


def test_transport_rejects_other_batch_even_when_session_and_revision_match(monkeypatch):
    from fdm.live_experiment.excel_template import ExcelTemplateError
    transport, process = transport_fixture(monkeypatch, [{'ok': True, 'input_written': True, 'batch_id': 'stale'}])
    with pytest.raises(ExcelTemplateError) as failure:
        transport.request('sync', 'trial-a', 5, 8, {'batch_id': 'current'})
    assert failure.value.code == 'protocol_mismatch'
    assert process.terminated


@pytest.mark.parametrize('in_flight', [False, True])
def test_parent_destruction_releases_worker_without_emitting_to_dead_qobject(tmp_path, monkeypatch, desktop_application, in_flight):
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QWidget
    from fdm.ui import live_excel
    template = tmp_path/'test.xlt'
    template.write_bytes(b'fixture')
    monkeypatch.setattr(live_excel,'working_template', lambda **kwargs: template)
    monkeypatch.setattr(live_excel,'template_preferences', lambda **kwargs: {})
    parent = QWidget()
    transport = FakeTransport()
    manager = live_excel.LiveExcelManager(parent, transport=transport, coalesce_ms=5)
    if in_flight:
        transport.gate.clear()
        manager.ensure(trial())
        wait_until(desktop_application, lambda: bool(transport.requests))
    parent.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    transport.gate.set()
    manager._thread.join(timeout=1)
    assert not manager._thread.is_alive()
    assert transport.closed


def test_excel_busy_finishes_fixed_batch_then_writes_latest_snapshot(coordinator):
    manager, transport, app = coordinator
    session = trial()
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session))
    transport.failure = {'ok':False,'error_code':'excel_busy','message':'正在编辑单元格','needs_full_sync':True}
    measurement_at(session, 12,'a')
    manager.update(session)
    wait_until(app, lambda: manager.status(session.id)['state'] == 'busy')
    measurement_at(session, 18,'a')
    manager.update(session)
    wait_until(app, lambda: manager.flush(session))
    requests = [item for item in transport.requests if item[0] == 'sync']
    assert requests[0][4] == requests[1][4]
    assert not requests[-1][4]['full_sync']
    assert requests[-1][4]['blocks'][0]['values'] == [[18]]
    assert requests[-1][4]['blocks'][0]['cell'] == 'D26'


def test_template_label_mapping_is_scoped_to_current_method(coordinator, monkeypatch):
    from fdm.ui import live_excel
    manager, transport, app = coordinator
    session = trial()
    session.method = 'gb16988'
    monkeypatch.setattr(live_excel, 'template_preferences', lambda **kwargs:
                        {'method_label': '旧全局覆盖不得使用', 'method_labels': {'gb16988':'自定义 GB 标签'}})
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session))
    first = [item for item in transport.requests if item[0] in ('open','sync')][-1]
    assert first[4]['report_settings'] is None
    assert manager._links[session.id].projection.data()['report_settings']['method_label'] == '自定义 GB 标签'
    session.method = 'iso17751'
    session.revision += 1
    manager.update(session)
    wait_until(app, lambda: manager.flush(session))
    assert len([item for item in transport.requests if item[0] in ('open','sync')]) == 1
    manager.update_report(session, information=False, settings=True)
    wait_until(app, lambda: any(item[0] == 'update_report' for item in transport.requests))
    last = [item for item in transport.requests if item[0] == 'update_report'][-1]
    assert last[4]['report_settings']['method_label'] == 'ISO 17751-1:2016(E)'


@pytest.mark.parametrize('explicit_selection', [False, True])
def test_template_method_labels_stay_with_this_book_after_defaults_change(coordinator, monkeypatch, explicit_selection):
    from fdm.ui import live_excel
    manager, transport, app = coordinator
    session = trial()
    session.method = 'gb16988'
    preferences = {'method_labels': {'gb16988': '当前工作簿的标准标签'}}
    monkeypatch.setattr(live_excel, 'template_preferences', lambda **kwargs: preferences)
    if explicit_selection:
        manager.replace_template(session, preferences)
    else:
        manager.ensure(session)
    wait_until(app, lambda: manager.flush(session))
    preferences = {'method_labels': {'gb16988': '另一个模板的不同标签'}}
    measurement_at(session, 12, 'a')
    manager.resync(session)
    wait_until(app, lambda: manager.flush(session))
    assert manager._links[session.id].projection.data()['report_settings']['method_label'] == '当前工作簿的标准标签'
    manager.update_report(session, information=False, settings=True)
    wait_until(app, lambda: any(item[0] == 'update_report' for item in transport.requests))
    updated = [item for item in transport.requests if item[0] == 'update_report'][-1]
    assert updated[4]['report_settings']['method_label'] == '当前工作簿的标准标签'
    wait_until(app, lambda: manager._busy is None)
    transport.failure = {'ok': False, 'error_code': 'workbook_closed'}
    manager.show(session.id)
    wait_until(app, lambda: manager.status(session.id)['state'] == 'stopped')
    manager.resync(session)
    wait_until(app, lambda: manager.flush(session))
    reopened = [item for item in transport.requests if item[0] == 'open'][-1]
    assert reopened[4]['report_settings'] is None
    assert manager.template_binding(session)['preferences']['method_labels']['gb16988'] == '当前工作簿的标准标签'


def test_status_uses_cached_template_identity_without_decoding_trial_inputs(coordinator, monkeypatch):
    manager, transport, app = coordinator
    session = trial()
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session))
    expected = manager.status(session.id)['template_path']
    value = manager._links[session.id].projection

    def no_decode(self):
        raise AssertionError('status must not decode the full trial input')

    monkeypatch.setattr(type(value), 'data', no_decode)
    assert manager.status(session.id)['template_path'] == expected


def test_status_poll_does_not_erase_failed_calculation_state(coordinator):
    manager, transport, app = coordinator
    session = trial()
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session))
    measurement_at(session, 14,'a')
    transport.failure = {'ok':False,'error_code':'excel_failed','message':'模板宏失败','needs_full_sync':True}
    manager.update(session)
    wait_until(app, lambda: manager.status(session.id)['state'] == 'error')
    manager.show(session.id)
    wait_until(app, lambda: manager._busy is None)
    assert manager.status(session.id)['state'] == 'error'
    assert manager.status(session.id)['unsynced']


def test_compact_controls_reuse_status_row_and_keep_menu_available(coordinator):
    from PySide6.QtWidgets import QLabel, QPushButton, QToolButton, QWidget
    from fdm.ui.live_excel import LiveExcelControls
    manager, transport, app = coordinator
    session = trial()
    parent = QWidget()
    label = QLabel(parent)
    label.resize(110, 25)
    controls = LiveExcelControls(manager, session, parent, compact=True, status_label=label)
    assert controls.layout().count() == 1
    assert controls.findChildren(QPushButton) == []
    assert len(controls.findChildren(QToolButton)) == 1
    assert label.parent() is parent
    assert not label.wordWrap()
    assert label.toolTip() == '进入试验后自动连接 Excel'
    assert controls.excel_button.isEnabled()
    assert not controls.show_button.isEnabled()
    assert controls.resync_button.isEnabled()
    assert controls.details_action.isEnabled()
    assert label.text() != label.toolTip()
    assert '\n' not in label.text()
    assert controls.excel_button.popupMode() == QToolButton.ToolButtonPopupMode.MenuButtonPopup
    assert controls.excel_button.minimumWidth() >= 80
    assert controls.excel_button.minimumWidth() >= controls.excel_button.fontMetrics().horizontalAdvance('Excel') + 28
    controls.details_action.trigger()
    app.processEvents()
    assert controls._details.text() == label.toolTip()
    controls._details.close()
    parent.close()


def test_compact_controls_actions_and_full_warning_details(coordinator):
    from PySide6.QtWidgets import QLabel
    from fdm.ui.live_excel import LiveExcelControls
    manager, transport, app = coordinator
    session = trial()
    label = QLabel()
    label.resize(130, 25)
    controls = LiveExcelControls(manager, session, compact=True, status_label=label)
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session))
    manager._links[session.id].metadata['warnings'] = ['超过模板统计范围 24 条，请复核。']
    manager.changed.emit(session.id)
    assert '超过模板统计范围 24 条' in label.toolTip()
    assert '\n' not in label.text()
    assert controls.show_button.isEnabled()
    controls.excel_button.click()
    wait_until(app, lambda: any(item[0] == 'show' for item in transport.requests))
    controls.stop_button.trigger()
    wait_until(app, lambda: manager.is_detached(session.id))
    assert controls.excel_button.isEnabled()  # Reconnect stays reachable.
    assert not controls.stop_button.isEnabled()
    controls.close()
    label.close()


@pytest.mark.parametrize('count,limited_excess,main_excess', [
    (1988, 0, 0), (1989, 1, 0), (2976, 988, 0), (2977, 989, 1)])
def test_diameter_report_specific_capacity_does_not_truncate_input(count, limited_excess, main_excess):
    session = trial('diameter')
    for index in range(count):
        measurement_at(session, 12 + index / 10000, 'a')
    request = template_request(ExportSnapshot.capture(session), 'sheet_dia.xlt', 'report.xls')
    assert sum(len(block['values']) for block in request['blocks']) == count
    limited = [warning for warning in request['warnings'] if '部分标准报告' in warning]
    main = [warning for warning in request['warnings'] if '直径模板统计范围' in warning]
    assert bool(limited) == bool(limited_excess)
    assert bool(main) == bool(main_excess)
    if limited:
        assert 'TWC-TM24-2009／ASTM-D-2130' in limited[0]
        assert f'可能少计 {limited_excess} 条' in limited[0]
        assert '全部原始数据仍完整写入' in limited[0]
    if main:
        assert f'可能少计 {main_excess} 条' in main[0]


def test_diameter_standard_report_capacity_is_configurable_and_validated(tmp_path):
    mapping = tmp_path/'mapping.json'
    mapping.write_text(json.dumps({'standard_report_capacity': 2500}))
    assert DiameterTemplateProfile.load(mapping).standard_report_capacity == 2500
    mapping.write_text(json.dumps({'standard_report_capacity': 0}))
    with pytest.raises(ValueError, match='standard_report_capacity'):
        DiameterTemplateProfile.load(mapping)


def test_diameter_comparisons_distinguish_raw_average_and_grouped_mean():
    session = trial('diameter')
    empty = template_request(ExportSnapshot.capture(session), 'sheet_dia.xlt', 'report.xls')
    assert empty['checks'][0]['raw_range'] is None
    assert empty['checks'][0]['local_mean'] is None
    measurement_at(session, 12.1, 'a')
    measurement_at(session, 15.2, 'a')
    request = template_request(ExportSnapshot.capture(session), 'sheet_dia.xlt', 'report.xls')
    check = request['checks'][0]
    assert check['raw_range'] == 'C25:C26'
    assert check['local_mean'] == pytest.approx(13.65)
    assert check['mean_cell'] == 'C9'
    assert check['mean_semantics'] == 'grouped'


def test_connected_diameter_controls_explain_template_grouped_mean(coordinator):
    from PySide6.QtWidgets import QLabel
    from fdm.ui.live_excel import LiveExcelControls
    manager, transport, app = coordinator
    session = trial('diameter')
    label = QLabel()
    label.resize(150, 25)
    controls = LiveExcelControls(manager, session, compact=True, status_label=label)
    assert '分组计算' not in label.toolTip()
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session))
    assert 'Excel 模板均值按分组计算；FDM 面板显示原始均值，两者可能不同。' in label.toolTip()
    controls._show_details()
    assert '分组计算' in controls._details.text()
    controls._details.close()
    manager.stop(session.id)
    wait_until(app, lambda: manager.is_detached(session.id))
    assert '分组计算' not in label.toolTip()
    controls.close()
    label.close()


def test_closed_workbook_status_is_chinese_with_original_exception_in_details(coordinator):
    from fdm.ui.live_excel import LiveExcelControls
    manager, transport, app = coordinator
    session = trial()
    controls = LiveExcelControls(manager, session, compact=True)
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session))
    transport.failure = {'ok': False, 'error_code': 'workbook_closed',
                         'message': 'The live workbook was closed. Use reconnect to open a new workbook.'}
    manager.show(session.id)
    wait_until(app, lambda: manager.status(session.id)['state'] == 'stopped')
    status = manager.status(session.id)
    assert status['message'] == '工作簿已关闭，重新同步可恢复。'
    assert status['raw_error'].startswith('The live workbook was closed.')
    controls._show_details()
    assert controls._details.text().startswith('工作簿已关闭 · 点击“打开 Excel”恢复同步')
    assert status['message'] in controls._details.text()
    assert controls.status_label.property('statusTone') == 'warning'
    assert 'The live workbook was closed.' in controls._details.detailedText()
    controls._details.close()
    controls.close()


def test_template_error_is_readable_and_success_clears_original_error(coordinator):
    manager, transport, app = coordinator
    session = trial()
    transport.failure = {'ok': False, 'error_code': 'template_method',
                         'message': 'Template standard is missing or ambiguous.'}
    manager.ensure(session)
    wait_until(app, lambda: manager.status(session.id)['state'] == 'error')
    assert '请检查方法映射' in manager.status(session.id)['message']
    assert manager.status(session.id)['raw_error'] == 'Template standard is missing or ambiguous.'
    manager.resync(session)
    wait_until(app, lambda: manager.flush(session))
    assert 'raw_error' not in manager.status(session.id)


def test_warm_synchronization_defers_status_until_idle(coordinator):
    manager, transport, app = coordinator
    session = trial()
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session))
    for value in (12, 13, 14):
        measurement_at(session, value, 'a')
        manager.update(session)
        wait_until(app, lambda: manager.flush(session))
        manager._pump()
        assert manager._links[session.id].next_poll > time.monotonic()
    assert [item[0] for item in transport.requests] == ['open', 'sync', 'sync', 'sync']
    manager._links[session.id].next_poll = time.monotonic() - .01
    manager._pump()
    wait_until(app, lambda: manager._busy is None)
    assert transport.requests[-1][0] == 'status'
    assert manager._links[session.id].next_poll > time.monotonic()
    manager._pump()
    assert [item[0] for item in transport.requests].count('status') == 1


def test_detach_poll_is_idempotent_preserves_busy_reason_and_can_only_upgrade_force(coordinator):
    manager, transport, app = coordinator
    session = trial()
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session))
    transport.failure = {'ok': False, 'error_code': 'excel_busy', 'message': 'Excel is editing a cell.'}
    assert not manager.detach(session)
    wait_until(app, lambda: manager.status(session.id)['state'] == 'busy')
    link = manager._links[session.id]
    assert link.detaching and not link.force_detach
    original_message = link.message
    changed = []
    manager.changed.connect(changed.append)
    assert not manager.detach(session)
    assert not manager.detach(session, force=True)
    assert link.force_detach
    assert not manager.detach(session, force=False)
    assert link.force_detach
    assert link.state == 'busy'
    assert link.message == original_message
    assert changed == []
    assert len([request for request in transport.requests if request[0] == 'detach']) == 1
    link.retry_at = time.monotonic() - .01
    assert not manager.detach(session)
    wait_until(app, lambda: manager.is_detached(session.id))
    assert len([request for request in transport.requests if request[0] == 'detach']) == 2
