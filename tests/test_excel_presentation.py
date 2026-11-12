"""Explicit Excel window intentions remain separate from quantitative ACKs."""
import pytest

from test_excel_live import coordinator, measurement_at, trial, wait_until


def completed(manager, sid):
    return manager._busy is None and not manager.status(sid).get('presentation_pending')


def test_single_click_initial_open_and_repeated_clicks_create_one_book(coordinator):
    manager, transport, app = coordinator
    session = trial()
    measurement_at(session, 12, 'a')
    transport.gate.clear()
    manager.present(session)
    wait_until(app, lambda: len(transport.requests) == 1)
    for _ in range(5):
        manager.present(session)
    assert manager.status(session.id)['presentation_pending']
    transport.gate.set()
    wait_until(app, lambda: completed(manager, session.id))
    assert [request[0] for request in transport.requests] == ['open', 'show']
    assert manager.status(session.id)['input_revision'] == session.revision


def test_stale_connected_click_recovers_once_with_complete_input(coordinator):
    manager, transport, app = coordinator
    session = trial()
    measurement_at(session, 12, 'a')
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session) and manager._busy is None)
    old_generation = manager.status(session.id)['generation']
    transport.requests.clear()
    transport.failure = {'ok': False, 'error_code': 'workbook_closed'}
    manager.present(session)
    wait_until(app, lambda: completed(manager, session.id))
    assert [request[0] for request in transport.requests] == ['show', 'open', 'show']
    assert transport.requests[1][4]['full_sync']
    assert transport.requests[1][4]['blocks'][0]['values'] == [[12]]
    assert manager.status(session.id)['generation'] == old_generation + 1
    assert manager.status(session.id)['presentation_status'] == 'shown'


def test_explicit_stop_requires_click_to_resume_and_resync_presents_after_ack(coordinator):
    manager, transport, app = coordinator
    session = trial()
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session) and manager._busy is None)
    manager.stop(session.id)
    wait_until(app, lambda: manager.is_detached(session.id))
    manager.ensure(session)
    transport.requests.clear()
    manager._pump()
    assert not transport.requests
    manager.present(session)
    wait_until(app, lambda: completed(manager, session.id))
    assert [r[0] for r in transport.requests] == ['open', 'show']
    transport.requests.clear()
    manager.resync(session, present_after=True)
    manager.resync(session, present_after=True)
    wait_until(app, lambda: completed(manager, session.id))
    assert [r[0] for r in transport.requests] == ['sync', 'show']
    assert 'report_cells' not in transport.requests[0][4]
    assert 'report_settings' not in transport.requests[0][4]


def test_cancel_during_open_keeps_sync_without_late_show(coordinator):
    manager, transport, app = coordinator
    session = trial()
    transport.gate.clear()
    manager.present(session)
    wait_until(app, lambda: bool(transport.requests))
    manager.cancel_presentation()
    transport.gate.set()
    wait_until(app, lambda: manager.flush(session) and manager._busy is None)
    assert [r[0] for r in transport.requests] == ['open']


def test_previous_show_completion_cannot_consume_new_resync_display_intent(coordinator):
    manager, transport, app = coordinator
    session = trial()
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session) and manager._busy is None)
    transport.requests.clear()
    transport.gate.clear()
    manager.show(session.id)
    wait_until(app, lambda: len(transport.requests) == 1)
    manager.resync(session, present_after=True)
    transport.gate.set()
    wait_until(app, lambda: completed(manager, session.id))
    assert [r[0] for r in transport.requests] == ['show', 'sync', 'show']
    assert transport.requests[-1][4]['presentation_request']


def test_cancel_after_busy_window_restore_prevents_late_activation(coordinator):
    manager, transport, app = coordinator
    session = trial()
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session) and manager._busy is None)
    transport.requests.clear()
    transport.failure = {'ok': True, 'activation_deferred': True, 'window_visible': True}
    manager.present(session)
    wait_until(app, lambda: manager.status(session.id).get('presentation_status') == 'deferred')
    manager.cancel_presentation(session.id)
    manager._pump()
    assert [r[0] for r in transport.requests] == ['show']
    assert manager.status(session.id).get('presentation_status') != 'deferred'
    assert manager.flush(session)


def test_latest_trial_intent_supersedes_pending_old_trial_show(coordinator):
    manager, transport, app = coordinator
    first, second = trial(), trial()
    transport.gate.clear()
    manager.present(first)
    wait_until(app, lambda: bool(transport.requests))
    manager.present(second)
    transport.gate.set()
    wait_until(app, lambda: completed(manager, second.id))
    assert [r[1] for r in transport.requests if r[0] == 'show'] == [second.id]
    assert manager.flush(first) and manager.flush(second)


def test_busy_show_restores_first_and_retries_activation_without_rewriting(coordinator):
    manager, transport, app = coordinator
    session = trial()
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session) and manager._busy is None)
    previous = manager._links[session.id].projection
    transport.requests.clear()
    transport.failure = {'ok': True, 'presentation_status': 'deferred',
                         'activation_deferred': True, 'window_visible': True, 'window_handle': 123}
    manager.present(session)
    wait_until(app, lambda: manager.status(session.id).get('presentation_status') == 'deferred')
    assert manager.status(session.id)['window_visible']
    assert manager.flush(session)
    manager._links[session.id].presentation['retry_at'] = 0
    wait_until(app, lambda: completed(manager, session.id))
    assert [r[0] for r in transport.requests] == ['show', 'show']
    assert transport.requests[0][4]['request_foreground'] is True
    assert transport.requests[1][4]['request_foreground'] is False
    assert manager._links[session.id].projection == previous


def test_foreground_denied_is_notice_not_data_error(coordinator):
    manager, transport, app = coordinator
    session = trial()
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session) and manager._busy is None)
    transport.failure = {'ok': True, 'presentation_status': 'attention_required',
                         'window_visible': True, 'foreground': False,
                         'presentation_message': '请点击任务栏中的 Excel'}
    manager.present(session)
    wait_until(app, lambda: completed(manager, session.id))
    status = manager.status(session.id)
    assert status['state'] == 'synced' and not status['unsynced']
    assert 'error_code' not in status


def test_newly_reopened_book_closing_again_does_not_loop(coordinator):
    manager, transport, app = coordinator
    session = trial()
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session) and manager._busy is None)
    original = transport.request

    def request(operation, *args, **kwargs):
        if operation == 'show':
            transport.failure = {'ok': False, 'error_code': 'workbook_closed'}
        return original(operation, *args, **kwargs)

    transport.request = request
    transport.requests.clear()
    manager.present(session)
    wait_until(app, lambda: completed(manager, session.id))
    assert [r[0] for r in transport.requests] == ['show', 'open', 'show']
    assert not manager.status(session.id)['connected']


def test_compact_primary_fixed_width_and_error_prefix(coordinator):
    from PySide6.QtWidgets import QLabel, QWidget
    from fdm.ui.live_excel import LiveExcelControls
    from fdm.ui.live_experiment_view import apply_trial_theme
    manager, transport, app = coordinator
    session = trial()
    parent, label = QWidget(), QLabel()
    controls = LiveExcelControls(manager, session, parent, compact=True, status_label=label)
    apply_trial_theme(parent)
    width = controls.excel_button.width()
    assert controls.excel_button.property('emphasis') == 'primary'
    assert not controls.excel_button.icon().isNull()
    manager.ensure(session)
    wait_until(app, lambda: manager.flush(session) and manager._busy is None)
    assert controls.excel_button.width() == width
    original_status = manager.status
    manager.status = lambda sid: {**original_status(sid), 'template_replacement_failed': True,
                                  'template_replacement_message': '输入工作表不存在',
                                  'pending_template_path': 'new.xlt'}
    manager.changed.emit(session.id)
    assert label.toolTip().startswith('模板更换失败 · 仍使用旧工作簿')
    assert label.property('statusTone') == 'error'
    assert '#acquisitionTrial QLabel[statusTone="error"]' in parent.styleSheet()
    assert controls.excel_button.isEnabled() and controls.resync_button.isEnabled()
    manager.status = original_status
    parent.close()
    label.close()


@pytest.mark.parametrize('tone', ['error', 'warning', 'info'])
@pytest.mark.parametrize('theme', ['dark', 'light'])
def test_trial_hint_semantic_color_survives_local_theme_and_is_painted(coordinator, tone, theme):
    from PySide6.QtGui import QColor, QPalette
    from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget
    from fdm.ui.live_experiment_view import apply_trial_theme
    from fdm.ui.theme import build_theme_palette, semantic_colors
    manager, transport, app = coordinator
    widget = QWidget()
    layout = QVBoxLayout(widget)
    label = QLabel('Excel ABC 888 异常等待', widget)
    label.setObjectName('trialHint')
    label.setProperty('statusTone', tone)
    layout.addWidget(label)
    original = app.palette()
    try:
        app.setPalette(build_theme_palette(theme))
        apply_trial_theme(widget)
        widget.resize(400, 80)
        widget.show()
        app.processEvents()
        target = QColor(getattr(semantic_colors(app.palette()), tone))
        assert label.palette().color(QPalette.ColorRole.WindowText) == target
        painted = label.grab().toImage()
        # Inspect actual text pixels, not only a dynamic property / stylesheet.
        assert sum(painted.pixelColor(x, y) == target for x in range(painted.width())
                   for y in range(painted.height())) > 8
    finally:
        widget.close()
        app.setPalette(original)


def test_configuration_failure_opens_details_without_retrying_automatically(coordinator):
    from fdm.ui.live_excel import LiveExcelControls
    manager, transport, app = coordinator
    session = trial()
    manager.ensure(session)
    transport.failure = {'ok': False, 'error_code': 'template_mapping'}
    wait_until(app, lambda: manager.status(session.id)['state'] == 'error')
    controls = LiveExcelControls(manager, session, compact=True)
    assert controls.excel_button.text() == '查看异常'
    before = len(transport.requests)
    controls.excel_button.click()
    app.processEvents()
    assert len(transport.requests) == before
    assert controls._details.isVisible()
    entered = []
    controls.templateSettingsRequested.connect(lambda: entered.append(controls._details.isVisible()))
    controls._template_details_button.click()
    assert entered == [False]  # QMessageBox closes before the page-selection slot.
    controls.close()


def test_automatic_connection_shows_progress_then_neutral_synced_and_stopped(coordinator):
    from fdm.ui.live_excel import LiveExcelControls
    manager, transport, app = coordinator
    session = trial()
    controls = LiveExcelControls(manager, session, compact=True)
    transport.gate.clear()
    manager.ensure(session)
    wait_until(app, lambda: bool(transport.requests))
    assert not manager.status(session.id)['presentation_pending']
    assert controls.excel_button.text() == '连接中…'
    assert controls.status_label.property('statusTone') == 'info'
    transport.gate.set()
    wait_until(app, lambda: manager.flush(session) and manager._busy is None)
    assert controls.excel_button.text() == '查看 Excel'
    assert controls.status_label.property('statusTone') == ''
    manager.stop(session.id)
    wait_until(app, lambda: manager.is_detached(session.id))
    assert controls.excel_button.text() == '恢复同步'
    assert controls.status_label.property('statusTone') == ''
    controls.close()
