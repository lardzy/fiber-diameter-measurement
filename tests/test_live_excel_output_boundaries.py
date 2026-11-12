"""Live Excel ownership, partial-write retries and visible input ACK boundaries."""
from dataclasses import replace
import threading
import time

import pytest

from fdm.live_experiment.excel_live import input_patch, report_update
from fdm.live_experiment.excel_template import ExportSnapshot
from fdm.ui.live_excel import LiveExcelControls, ReportInformationDialog
from test_excel_live import coordinator, measurement_at, projection, trial, wait_until


@pytest.mark.parametrize('mode', ['content', 'diameter'])
def test_report_information_and_settings_are_not_quantitative_patches(mode):
    session = trial(mode)
    measurement_at(session, 12, 'a')
    before = projection(session)
    session.sample_name, session.sample_number = '用户新样品', 'N-007'
    session.method, session.group_width, session.grouped = 'iso17751', 2, False
    next(iter(session.operators.values())).name = '新人员'
    session.revision += 1
    after = projection(session)
    assert input_patch(before, after) is None
    full = input_patch(None, after)
    report = report_update(after, information=True, settings=True)
    report_cells = {cell['cell'] for cell in report['report_cells']}
    assert all(cell['cell'] not in report_cells for cell in full['cells'])
    assert all(item['range'] not in report_cells for item in full['clears'])
    assert 'report_settings' not in full and 'report_cells' not in full
    assert 'checks' not in full and 'managed_ranges' not in full
    assert any(cell['value'] == '用户新样品' for cell in report['report_cells'])
    assert report['report_settings']['group_width'] == 2
    assert not report['report_settings']['grouped']


def test_diameter_edits_write_contiguous_changed_ranges_only():
    session = trial()
    records = [measurement_at(session, value, 'a', f'2026-11-11T00:00:{index:02d}+00:00')
               for index, value in enumerate((10, 20, 30, 40, 50))]
    before = projection(session)
    for record, value in zip(records[1:3] + records[4:], (21, 31, 51)):
        session.measurements[record.id] = replace(record, diameter_um=value)
    after = projection(session)
    patch = input_patch(before, after)
    assert patch['blocks'] == [
        {'sheet': '原始数据', 'cell': 'D26', 'values': [[21], [31]]},
        {'sheet': '原始数据', 'cell': 'D29', 'values': [[51]]}]
    assert not patch['clears'] and not patch['cells']


def test_equal_timestamp_record_order_is_stable_after_delete_and_restore():
    session = trial()
    records = [session.add_measurement(value, 'a') for value in (10, 20, 30)]
    for record in records:
        session.measurements[record.id] = replace(record, created_at='2026-11-11T00:00:00+00:00')
    before = projection(session)
    restored = session.measurements.pop(records[0].id)
    session.measurements[restored.id] = restored
    assert projection(session) == before
    expected = [record.diameter_um for record in sorted(session.measurements.values(), key=lambda item: item.id)]
    assert before.data()['raw']['D'] == expected


def test_undo_returns_deleted_record_to_original_excel_row():
    session = trial()
    records = [measurement_at(session, value, 'a', f'2026-11-11T00:00:{index:02d}+00:00')
               for index, value in enumerate((10, 20, 30))]
    before = projection(session)
    removed = session.measurements.pop(records[0].id)
    partial = projection(session)
    session.measurements[removed.id] = removed
    assert input_patch(before, projection(session)) is None
    restored = input_patch(partial, projection(session))
    assert restored['blocks'][0]['cell'] == 'D25'
    assert restored['blocks'][0]['values'] == [[10], [20], [30]]


def test_append_requires_matching_record_prefix_not_just_equal_diameters():
    session = trial()
    old = measurement_at(session, 10, 'a', '2026-11-11T00:00:00+00:00')
    before = projection(session)
    session.measurements.pop(old.id)
    measurement_at(session, 10, 'a', '2026-11-11T00:00:01+00:00')
    measurement_at(session, 20, 'a', '2026-11-11T00:00:02+00:00')
    patch = input_patch(before, projection(session))
    assert patch['blocks'][0]['cell'] == 'D25'
    assert patch['blocks'][0]['values'] == [[10], [20]]


def test_capacity_warning_does_not_infer_excel_settings_from_fdm_grouped_flag():
    session = trial()
    session.grouped = False
    for _ in range(1489):
        measurement_at(session, 10, 'a')
    data = projection(session).data()
    assert not data['report_settings']['grouped']
    assert any('1488' in warning for warning in data['warnings'])


def test_partial_write_busy_then_undo_does_not_leave_unacknowledged_value(coordinator):
    manager, transport, app = coordinator
    session = trial()
    record = measurement_at(session, 10, 'a')
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session))
    session.measurements[record.id] = replace(record, diameter_um=20)
    session.revision += 1
    # Simulates Excel already receiving D25=20 before a later operation is busy.
    transport.failure = {'ok': False, 'error_code': 'excel_busy', 'completed_ops': 1}
    manager.update(session)
    wait_until(app, lambda: manager.status(session.id)['state'] == 'busy')
    first = [item for item in transport.requests if item[0] == 'sync'][-1]
    session.measurements[record.id] = record
    session.revision += 1
    manager.update(session)
    manager._links[session.id].retry_at = 0
    wait_until(app, lambda: manager.flush(session))
    requests = [item for item in transport.requests if item[0] == 'sync']
    assert requests[0][4] == requests[1][4] == first[4]
    assert requests[2][4]['batch_id'] != first[4]['batch_id']
    assert requests[2][4]['blocks'] == [{'sheet': '原始数据', 'cell': 'D25', 'values': [[10]]}]
    assert not requests[2][4]['full_sync']


def test_auxiliary_busy_cannot_discard_failed_partial_quantitative_batch(coordinator):
    manager, transport, app = coordinator
    session = trial()
    record = measurement_at(session, 10, 'a')
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session))
    session.measurements[record.id] = replace(record, diameter_um=20)
    session.revision += 1
    transport.failure = {'ok': False, 'error_code': 'excel_failed', 'completed_ops': 1}
    manager.update(session)
    wait_until(app, lambda: manager.status(session.id)['state'] == 'error')
    link = manager._links[session.id]
    batch_id = link.pending_batch.payload['batch_id']
    session.measurements[record.id] = record
    session.revision += 1
    transport.failure = {'ok': False, 'error_code': 'excel_busy'}
    manager.recalculate(session)
    wait_until(app, lambda: manager.status(session.id)['state'] == 'busy')
    assert link.pending_batch.payload['batch_id'] == batch_id
    assert link.pending_auxiliary.operation == 'recalculate'
    link.retry_at = 0
    manager.update(session)
    wait_until(app, lambda: manager.flush(session))
    inputs = [item for item in transport.requests if item[0] == 'sync']
    assert inputs[0][4] == inputs[1][4]
    assert inputs[-1][4]['blocks'][0]['values'] == [[10]]
    report = [item for item in transport.requests if item[0] == 'recalculate']
    assert report[0][4] == report[1][4]


@pytest.mark.parametrize('operation', ['show', 'status'])
@pytest.mark.parametrize('code', ['excel_busy', 'excel_failed'])
def test_readonly_failure_does_not_invalidate_input_or_force_full_output(coordinator, operation, code):
    manager, transport, app = coordinator
    session = trial()
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session))
    link = manager._links[session.id]
    previous = link.projection
    transport.failure = {'ok': False, 'error_code': code, 'message': '不能对Null值表达式调用方法',
                         'stage': 'read workbook saved', 'error_stack': 'fixture line 123'}
    if operation == 'show':
        manager.show(session.id)
    else:
        link.next_poll = 0
        manager._pump()
    wait_until(app, lambda: manager._busy is None)
    assert link.projection == previous
    assert not link.dirty and not link.force_full and link.pending_batch is None
    assert manager.flush(session) and not manager.status(session.id)['unsynced']
    assert manager.status(session.id)['save_state'] == 'unknown'
    assert manager.status(session.id)['saved'] is None
    assert manager.status(session.id)['error_stack'] == 'fixture line 123'
    measurement_at(session, 17, 'a')
    manager.update(session)
    link.retry_at = 0
    wait_until(app, lambda: manager.flush(session))
    last = [item for item in transport.requests if item[0] == 'sync'][-1][4]
    assert not last['full_sync']
    assert last['blocks'][0]['values'] == [[17]]
    assert not last['cells'] and not last['clears']


def test_visible_input_ack_finishes_lifecycle_before_metadata_and_survives_error(coordinator, monkeypatch):
    manager, transport, app = coordinator
    session = trial()
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session))
    gate = threading.Event()
    original = transport.request

    def request(operation, session_id='', generation=0, input_revision=0, payload=None, *, progress=None):
        if operation != 'sync':
            return original(operation, session_id, generation, input_revision, payload, progress=progress)
        progress({'event': 'inputs_written', 'input_written': True, 'session_id': session_id,
                  'generation': generation, 'input_revision': input_revision, 'batch_id': payload['batch_id']})
        gate.wait(3)
        return {'ok': False, 'error_code': 'excel_failed', 'message': 'metadata unavailable', 'stage': 'metadata'}

    monkeypatch.setattr(transport, 'request', request)
    measurement_at(session, 22, 'a')
    manager.update(session)
    try:
        wait_until(app, lambda: manager.status(session.id)['input_revision'] == session.revision)
        assert manager._busy is not None
        assert manager.flush(session)
        assert not manager.status(session.id)['unsynced']
    finally:
        gate.set()
    wait_until(app, lambda: manager._busy is None)
    assert manager.flush(session)
    assert not manager._links[session.id].dirty
    assert manager._links[session.id].pending_batch is None
    assert manager.status(session.id)['save_state'] == 'unknown'
    monkeypatch.setattr(transport, 'request', original)
    measurement_at(session, 23, 'a')
    manager.update(session)
    wait_until(app, lambda: manager.flush(session))
    assert transport.requests[-1][4]['blocks'][0]['values'] == [[23]]


def test_same_revision_full_output_waits_for_its_own_ack(coordinator):
    manager, transport, app = coordinator
    session = trial()
    measurement_at(session, 22, 'a')
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session))

    ack_revision = manager.status(session.id)['input_revision']
    transport.gate.clear()
    try:
        manager.resync(session)
        wait_until(app, lambda: any(item[0] == 'sync' for item in transport.requests))
        assert manager.status(session.id)['input_revision'] == ack_revision == session.revision
        assert not manager.flush(session)
        assert manager.status(session.id)['input_pending']
        assert not manager.detach(session)
    finally:
        transport.gate.set()
    wait_until(app, lambda: manager.flush(session))


@pytest.mark.parametrize('code', ['excel_failed', 'workbook_closed'])
def test_post_ack_failure_recovers_readonly_status_or_keeps_closed_state(coordinator, monkeypatch, code):
    manager, transport, app = coordinator
    session = trial()
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session))
    original = transport.request

    def request(operation, session_id='', generation=0, input_revision=0, payload=None, *, progress=None):
        if operation != 'sync':
            return original(operation, session_id, generation, input_revision, payload, progress=progress)
        progress({'event': 'inputs_written', 'input_written': True, 'session_id': session_id,
                  'generation': generation, 'input_revision': input_revision, 'batch_id': payload['batch_id']})
        return {'ok': False, 'error_code': code, 'message': 'post-input metadata failure'}

    monkeypatch.setattr(transport, 'request', request)
    measurement_at(session, 22, 'a')
    manager.update(session)
    wait_until(app, lambda: manager.status(session.id).get('error_code') == code)
    assert manager.flush(session)
    assert not manager.status(session.id)['unsynced']
    if code == 'workbook_closed':
        assert manager.status(session.id)['state'] == 'stopped'
        assert not manager.status(session.id)['connected']
        assert not manager.status(session.id)['active']
    else:
        manager.show(session.id)
        wait_until(app, lambda: manager._busy is None)
        assert 'error_code' not in manager.status(session.id)
        assert manager.status(session.id)['message'] == '测量数据已输出；报表由 Excel 计算'


def test_manual_excel_calculation_does_not_hold_final_input_barrier(coordinator):
    manager, transport, app = coordinator
    session = trial()
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session))
    link = manager._links[session.id]
    link.metadata.update(calculation_state='manual', save_state='unsaved_changes', saved=False)
    controls = LiveExcelControls(manager, session, compact=True)
    assert 'Excel 手动计算' in controls._full_status
    assert '文件有未保存修改' in controls._full_status
    assert manager.flush(session)
    assert not manager.status(session.id)['unsynced']
    controls.close()


def test_reoutput_preserves_report_fields_but_explicit_actions_choose_them(coordinator):
    manager, transport, app = coordinator
    session = trial()
    session.sample_name = '初始名称'
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session))
    initial = transport.requests[0][4]
    assert any(item['value'] == '初始名称' for item in initial['report_cells'])
    assert initial['report_settings'] is None
    session.sample_name, session.group_width = '后续名称', 3
    session.revision += 1
    manager.update(session)
    before = len(transport.requests)
    wait_until(app, lambda: manager.flush(session))
    assert len(transport.requests) == before
    manager.resync(session)
    wait_until(app, lambda: manager.flush(session))
    full = transport.requests[-1][4]
    assert full['full_sync']
    assert 'report_cells' not in full and 'report_settings' not in full
    manager.update_report(session, information=True, settings=False)
    wait_until(app, lambda: manager._busy is None)
    info = transport.requests[-1][4]
    assert info['report_settings'] is None
    assert any(item['value'] == '后续名称' for item in info['report_cells'])
    manager.recalculate(session)
    wait_until(app, lambda: manager._busy is None)
    assert set(transport.requests[-1][4]) == {'profile', 'batch_id'}
    assert not manager.status(session.id)['unsynced']


def test_explicit_report_snapshot_is_not_replaced_by_later_input_revision(coordinator):
    manager, transport, app = coordinator
    session = trial()
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session))
    session.sample_name = '用户确认的名称'
    confirmed = ExportSnapshot.capture(session)
    session.sample_name = '确认后另外变更'
    session.revision += 1
    manager.update_report(session, snapshot=confirmed)
    wait_until(app, lambda: manager._busy is None)
    report = [item for item in transport.requests if item[0] == 'update_report'][-1][4]
    assert any(cell['value'] == '用户确认的名称' for cell in report['report_cells'])
    assert not any(cell['value'] == '确认后另外变更' for cell in report['report_cells'])


def test_report_actions_use_open_workbook_mapping_when_template_file_is_missing(coordinator, monkeypatch):
    from fdm.ui import live_excel
    manager, transport, app = coordinator
    session = trial()
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session))
    monkeypatch.setattr(live_excel, 'working_template', lambda **kwargs: (_ for _ in ()).throw(FileNotFoundError('moved template')))
    manager.recalculate(session)
    wait_until(app, lambda: manager._busy is None)
    assert transport.requests[-1][0] == 'recalculate'
    manager.update_report(session)
    wait_until(app, lambda: manager._busy is None)
    assert transport.requests[-1][0] == 'update_report'
    assert manager.flush(session)


def test_nonbusy_partial_initialization_can_be_retried_in_same_workbook(coordinator):
    manager, transport, app = coordinator
    session = trial()
    transport.failure = {'ok': False, 'error_code': 'template_macro', 'workbook_connected': True}
    manager.ensure(session)
    wait_until(app, lambda: manager.status(session.id)['state'] == 'error')
    old_generation = manager.status(session.id)['generation']
    assert manager.status(session.id)['connected']
    manager.resync(session)
    wait_until(app, lambda: manager.flush(session))
    calls = [item for item in transport.requests if item[0] == 'open']
    assert len(calls) == 2
    assert calls[1][2] == old_generation
    assert calls[1][4]['report_settings'] is None


def test_report_failure_does_not_block_later_quantitative_input(coordinator):
    manager, transport, app = coordinator
    session = trial()
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session))
    transport.failure = {'ok': False, 'error_code': 'macro_failed'}
    manager.recalculate(session)
    wait_until(app, lambda: manager._busy is None)
    assert manager.flush(session)
    assert not manager._links[session.id].dirty
    assert manager._links[session.id].pending_batch is None
    measurement_at(session, 25, 'a')
    manager.update(session)
    wait_until(app, lambda: manager.flush(session))
    assert transport.requests[-1][0] == 'sync'
    assert manager.status(session.id)['report_error']
    manager.recalculate(session)
    wait_until(app, lambda: manager._busy is None)
    assert 'report_error' not in manager.status(session.id)


def test_report_update_dialog_defaults_and_exact_menu_labels(coordinator):
    manager, transport, app = coordinator
    session = trial()
    dialog = ReportInformationDialog(session)
    assert dialog.information.isChecked() and not dialog.settings.isChecked()
    dialog.information.setChecked(False)
    assert not dialog.apply_button.isEnabled()
    dialog.settings.setChecked(True)
    assert dialog.apply_button.isEnabled()
    controls = LiveExcelControls(manager, session, compact=True)
    labels = [action.text() for action in controls.excel_button.menu().actions()]
    assert labels[:5] == ['查看 Excel', '重新输出测量数据', '更新报表信息…', '重新统计报表', '停止同步']
    dialog.close()
    controls.close()


def test_unknown_save_status_never_claims_file_unsaved(coordinator):
    manager, transport, app = coordinator
    session = trial()
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session))
    link = manager._links[session.id]
    link.metadata.update(saved=None, save_state='unknown')
    controls = LiveExcelControls(manager, session, compact=True)
    assert '文件保存状态暂不可知' in controls._full_status
    assert '文件尚未保存' not in controls._full_status
    controls.close()
