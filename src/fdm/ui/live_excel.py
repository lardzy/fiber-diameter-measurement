"""Application-owned live Excel links; page widgets only display/control links."""
from __future__ import annotations

from dataclasses import dataclass, field
from queue import Queue
import threading
import time

from PySide6.QtCore import QEvent, QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QMenu, QMessageBox, QPushButton,
                               QSizePolicy, QStyle, QToolButton, QVBoxLayout, QWidget)

from fdm.live_experiment.excel_template import (ExportSnapshot, TemplateProfile,
                                               template_preferences, working_template)
from fdm.live_experiment.excel_live import PowerShellLiveTransport, input_patch, project_inputs


_EXCEL_ERROR_MESSAGES = {
    'workbook_closed': '工作簿已关闭，重新同步可恢复。',
    'excel_busy': 'Excel 正忙，恢复后将继续同步最新数据。',
    'template_missing': '找不到 Excel 模板，请在报告页选择模板后重新同步。',
    'template_method': 'Excel 模板缺少当前试验方法，或存在重复匹配；请检查方法映射。',
    'macro_failed': 'Excel 模板宏运行失败，请检查宏权限和模板后重新同步。',
    'calculation_timeout': 'Excel 计算超时，报表尚未确认完成；请检查后重新同步。',
    'worker_timeout': 'Excel 长时间无响应，最后一批数据状态未知；请检查后重新同步。',
    'worker_disconnected': 'Excel 同步连接已断开，请重新同步。',
    'worker_exited': 'Excel 同步组件已退出，请重新同步。',
    'excel_not_independent': '无法建立独立的 Excel 试验实例，请检查 Excel 后重试。',
    'template_protected': 'Excel 模板输入区受保护，无法建立同步；请检查模板。',
    'initialization_failed': 'Excel 输入区初始化未完成，请重新同步。',
    'stale_generation': 'Excel 工作簿关联已变化，未确认本次写入；请重新同步。',
    'stale_revision': 'Excel 拒绝了过期的试验数据，请重新同步最新记录。',
    'protocol_mismatch': 'Excel 返回的试验身份或修订不匹配，未确认写入；请重新同步。',
    'excel_failed': 'Excel 同步失败，请查看同步详情并重试。',
}


@dataclass
class _Link:
    session: object
    generation: int = 1
    state: str = 'connecting'
    message: str = '正在连接 Excel'
    active: bool = True
    connected: bool = False
    ever_connected: bool = False
    force_full: bool = False
    retry_at: float = 0.0
    reopen: bool = False
    workbook_projection: object = None
    dirty: bool = True
    acknowledged_revision: int = -1
    projection: object = None
    desired_action: str = ''
    detaching: bool = False
    force_detach: bool = False
    next_poll: float = 0.0
    metadata: dict = field(default_factory=dict)


class LiveExcelManager(QObject):
    changed = Signal(str)
    _finished = Signal(object)
    _progressed = Signal(object)

    def __init__(self, parent=None, *, transport=None, coalesce_ms=100):
        super().__init__(parent)
        self._transport = transport or PowerShellLiveTransport()
        self._links = {}
        self._busy = None
        self._closing = False
        self._closed = False
        self._shutdown_retry = 0.0
        self.close_message = ""
        self._tasks = Queue()
        self._destroyed_event = threading.Event()
        # Parent destruction can bypass the normal window-close flow (tests,
        # failed construction). Wake the daemon without touching a dead Qt object.
        tasks, destroyed_event = self._tasks, self._destroyed_event
        def release_on_destroyed(*_):
            destroyed_event.set()
            tasks.put(None)
        self.destroyed.connect(release_on_destroyed)
        self._finished.connect(self._complete)
        self._progressed.connect(self._progress)
        self._thread = threading.Thread(target=self._run, name='excel-live-sta-client', daemon=True)
        self._thread.start()
        self._timer = QTimer(self)
        self._timer.setInterval(coalesce_ms)
        self._timer.timeout.connect(self._pump)
        self._timer.start()

    def _run(self):
        while True:
            task = self._tasks.get()
            if task is None or self._destroyed_event.is_set():
                try:
                    self._transport.close()
                except Exception:
                    pass
                return
            sid, generation, operation, revision, snapshot, previous, force_full = task
            used_generation = generation
            projection = previous
            try:
                if operation == 'shutdown':
                    self._transport.close()
                    result = {'ok': True, 'status': 'closed'}
                elif operation in ('open', 'sync'):
                    mode = snapshot.trial_mode
                    preferences = template_preferences(trial_mode=mode)
                    profile = TemplateProfile.load(preferences.get('mapping'), trial_mode=mode)
                    template = working_template(create=True, trial_mode=mode)
                    if not template.is_file():
                        raise FileNotFoundError('试验 Excel 模板不存在，请在报告页选择模板后重新同步。')
                    method_labels = preferences.get('method_labels', {})
                    method_label = method_labels.get(snapshot.method) if isinstance(method_labels, dict) else None
                    projection = project_inputs(snapshot, template, profile, method_label=method_label)
                    if previous:
                        old, current = previous.data(), projection.data()
                        if any(old[key] != current[key] for key in ('template', 'profile', 'trial_kind')):
                            operation = 'open'
                            used_generation += 1
                            force_full = True
                    payload = input_patch(None if force_full else previous, projection)
                    if payload is None:
                        result = {'ok': True, 'status': 'synced', 'unchanged': True}
                    else:
                        result = self._transport.request(operation, sid, used_generation, revision, payload,
                            progress=lambda value: self._emit_progress(value))
                else:
                    result = self._transport.request(operation, sid, generation, revision)
            except Exception as exc:
                result = {'ok': False, 'error_code': getattr(exc, 'code', 'excel_failed'),
                          'message': str(exc), 'needs_full_sync': True}
            if self._destroyed_event.is_set():
                try:
                    self._transport.close()
                except Exception:
                    pass
                return
            try:
                self._finished.emit((sid, generation, operation, revision, projection, result, used_generation))
            except RuntimeError:
                # QObject can be destroyed between the event check and emit.
                try:
                    self._transport.close()
                except Exception:
                    pass
                return
            if operation == 'shutdown' and result.get('ok'):
                return

    def _emit_progress(self, value):
        if not self._destroyed_event.is_set():
            try:
                self._progressed.emit(value)
            except RuntimeError:
                self._destroyed_event.set()

    @staticmethod
    def _sid(session):
        return session if isinstance(session, str) else session.id

    def ensure(self, session):
        if self._closing:
            return
        if session.id not in self._links:
            self._links[session.id] = _Link(session)
            self.changed.emit(session.id)
        else:
            self.update(session)
        # Explicitly stopped, closed or failed links need a user resync. Reentry
        # does not recreate a workbook that the user deliberately closed.

    def update(self, session):
        link = self._links.get(session.id)
        if not link:
            return
        link.session = session
        if link.active and session.revision != link.acknowledged_revision:
            link.dirty = True

    def status(self, session_id):
        link = self._links.get(self._sid(session_id))
        if not link:
            return {'state': 'idle', 'message': '进入试验后自动连接 Excel', 'active': False,
                    'connected': False, 'pending': False, 'unsynced': False}
        pending = bool((link.active and link.dirty) or link.desired_action or link.detaching or
                       (self._busy and self._busy[0] == link.session.id))
        unsynced = bool(link.ever_connected and (link.dirty or link.projection is None or
                        link.session.revision != link.acknowledged_revision))
        return {**link.metadata, 'state': link.state, 'message': link.message, 'active': link.active,
                'connected': link.connected, 'ever_connected': link.ever_connected, 'pending': pending, 'unsynced': unsynced,
                'input_revision': link.acknowledged_revision, 'target_revision': link.session.revision,
                'generation': link.generation}

    def show(self, session_id):
        link = self._links.get(self._sid(session_id))
        if link and link.connected and not link.detaching:
            link.desired_action = 'show'
            self._pump()

    def resync(self, session):
        sid = self._sid(session)
        if isinstance(session, str):
            link = self._links.get(sid)
            if not link:
                return
            session = link.session
        if sid not in self._links:
            self.ensure(session)
            return
        link = self._links[sid]
        link.session = session
        link.active = True
        link.detaching = link.force_detach = False
        link.force_full = True
        link.dirty = True
        link.state, link.message = 'connecting', '正在重新同步完整试验'
        # Keep generation for the same known workbook; a manually closed book
        # gets a new generation. Never allow an old reply to confirm this one.
        if not link.connected and not (self._busy and self._busy[0] == sid):
            link.generation += 1
        link.desired_action = ''
        link.retry_at = 0.0
        if link.metadata.get('error_code') in ('initialization_failed', 'template_protected'):
            link.reopen = True
            link.generation += 1
        self.changed.emit(sid)

    def stop(self, session_id):
        link = self._links.get(self._sid(session_id))
        if link:
            self.detach(link.session, force=True)

    def flush(self, session):
        link = self._links.get(session.id)
        if not link:
            return True
        self.update(session)
        in_flight = bool(self._busy and self._busy[0] == session.id and self._busy[2] in ('open', 'sync'))
        if not link.ever_connected and not in_flight and link.state in ('error', 'unsupported', 'stopped', 'detached'):
            return True
        if link.state == 'detached':
            return True
        ready = (link.projection is not None and not link.dirty and
                 link.acknowledged_revision == session.revision and
                 not (self._busy and self._busy[0] == session.id and self._busy[2] in ('open', 'sync')))
        if not ready:
            self._pump()
        return ready

    ready_to_detach = flush

    def detach(self, session, *, force=False):
        link = self._links.get(self._sid(session))
        if not link or link.state == 'detached':
            return True
        force = force or link.force_detach
        if link.detaching:
            # Lifecycle dialogs poll this method. Do not erase a busy/error
            # explanation or emit a fresh transition on every poll.
            link.force_detach = force
            self._pump()
            return link.state == 'detached'
        if not isinstance(session, str):
            self.update(session)
        if not force and not self.flush(link.session):
            return False
        link.active = False
        link.detaching = True
        link.force_detach = force
        link.desired_action = ''
        link.state, link.message = 'detaching', '正在解除 Excel 关联，工作簿将保留在 Excel 中'
        self.changed.emit(link.session.id)
        self._pump()
        return link.state == 'detached'

    def is_detached(self, session_id):
        link = self._links.get(self._sid(session_id))
        return not link or link.state == 'detached'

    def close(self):
        if self._closed:
            return True
        self._closing = True
        for link in self._links.values():
            if link.state != 'detached':
                self.detach(link.session, force=True)
        self._pump()
        return self._closed

    def _dispatch(self, sid, link, operation, snapshot=None):
        generation = link.generation if link else 0
        revision = snapshot.revision if snapshot else (link.acknowledged_revision if link else 0)
        self._busy = sid, generation, operation, revision
        previous = (link.projection or link.workbook_projection) if link else None
        force_full = bool(link and link.force_full)
        if link and operation in ('open', 'sync'):
            link.force_full = False
        self._tasks.put((sid, generation, operation, revision, snapshot, previous, force_full))

    def _pump(self):
        if self._busy or self._closed:
            return
        now = time.monotonic()
        for sid, link in self._links.items():
            if link.detaching:
                if now < link.retry_at:
                    continue
                if not link.connected:
                    link.detaching = False
                    link.state, link.message = 'detached', '同步已停止；Excel 工作簿由用户保存'
                    self.changed.emit(sid)
                    continue
                self._dispatch(sid, link, 'detach')
                return
            if link.active and link.dirty and now >= link.retry_at and link.state not in ('error', 'stopped', 'unsupported'):
                try:
                    snapshot = ExportSnapshot.capture(link.session)
                except Exception as exc:
                    link.state, link.message = 'error', f'无法准备 Excel 数据：{exc}'
                    self.changed.emit(sid)
                    continue
                link.dirty = False
                link.state, link.message = 'syncing' if link.connected else 'connecting', '正在同步 Excel' if link.connected else '正在连接 Excel'
                self._dispatch(sid, link, 'sync' if link.connected and not link.reopen else 'open', snapshot)
                link.reopen = False
                self.changed.emit(sid)
                return
            if link.connected and link.desired_action:
                operation, link.desired_action = link.desired_action, ''
                self._dispatch(sid, link, operation)
                return
        if self._closing:
            if now >= self._shutdown_retry and all(link.state == 'detached' for link in self._links.values()):
                self._dispatch('', None, 'shutdown')
            return
        for sid, link in self._links.items():
            if link.connected and link.active and now >= link.next_poll:
                link.next_poll = now + 2.0
                self._dispatch(sid, link, 'status')
                return

    def _progress(self, value):
        sid = value.get('session_id')
        link = self._links.get(sid)
        if (not link or not self._busy or self._busy[0] != sid
                or value.get('generation') != link.generation
                or value.get('input_revision') != self._busy[3]):
            return
        if value.get('event') == 'inputs_written':
            link.metadata['inputs_written_revision'] = value['input_revision']
            link.metadata['write_elapsed_ms'] = value.get('elapsed_ms')
            link.state, link.message = 'calculating', '原始数据已写入，Excel 正在计算报表'
            self.changed.emit(sid)

    def _complete(self, value):
        sid, generation, operation, revision, projection, result, used_generation = value
        self._busy = None
        if operation == 'shutdown':
            self._closed = bool(result.get('ok'))
            self.close_message = '' if self._closed else _EXCEL_ERROR_MESSAGES.get(result.get('error_code'),
                result.get('message', 'Excel 关联尚未释放'))
            if self._closed:
                self._timer.stop()
            else:
                self._shutdown_retry = time.monotonic() + .5
            return
        link = self._links.get(sid)
        if not link or generation != link.generation:
            self._pump()
            return
        link.generation = used_generation
        if result.get('ok'):
            if operation in ('open', 'sync', 'show', 'status') and not result.get('unchanged'):
                # Successful replies already include current workbook metadata.
                # Poll only after two idle seconds, not between warm writes.
                link.next_poll = time.monotonic() + 2.0
            for key in ('saved', 'path', 'name', 'excel_version', 'excel_pid', 'warnings', 'comparisons', 'elapsed_ms'):
                if key in result:
                    link.metadata[key] = result[key]
            if operation in ('open', 'sync'):
                link.connected = link.ever_connected = True
                link.metadata.pop('raw_error', None)
                link.metadata.pop('error_code', None)
                link.projection = link.workbook_projection = projection
                link.acknowledged_revision = revision
                link.dirty = link.session.revision != revision or link.force_full
                link.state, link.message = 'synced', '已同步到 Excel' if not link.dirty else '正在等待下一批同步'
            elif operation == 'detach':
                link.connected = link.active = link.detaching = False
                link.state, link.message = 'detached', '同步已停止；Excel 工作簿由用户保存'
            elif not link.detaching and link.state not in ('error', 'busy', 'stopped', 'unsupported'):
                link.state, link.message = 'synced', '已同步到 Excel'
        else:
            code = result.get('error_code', result.get('code', 'excel_failed'))
            link.metadata['error_code'] = code
            link.state = 'unsupported' if code == 'unsupported_platform' else 'stopped' if code == 'workbook_closed' else 'error'
            raw_error = result.get('message') or result.get('error') or ''
            link.metadata['raw_error'] = raw_error
            # Keep useful locally generated Chinese diagnostics; translate the
            # stable worker code without discarding the original COM exception.
            link.message = (raw_error if code == 'excel_failed' and any('\u4e00' <= ch <= '\u9fff' for ch in raw_error)
                            else _EXCEL_ERROR_MESSAGES.get(code, raw_error or 'Excel 同步中断，请重试。'))
            link.dirty = True
            if result.get('name') or result.get('workbook_connected'):
                link.connected = link.ever_connected = True
                if projection is not None:
                    link.workbook_projection = projection
            if result.get('needs_full_sync'):
                link.projection = None
            if code in ('workbook_closed', 'worker_timeout', 'worker_exited', 'worker_disconnected', 'protocol_mismatch', 'unsupported_platform'):
                link.connected = False
                link.active = False
            if code == 'excel_busy':
                link.state, link.message = 'busy', 'Excel 正忙，恢复后将继续同步最新数据'
                link.retry_at = time.monotonic() + .5
                link.force_full = True
            elif operation == 'detach':
                link.detaching = False
            if code in ('worker_timeout', 'worker_exited', 'worker_disconnected', 'protocol_mismatch'):
                for other in self._links.values():
                    other.connected = other.active = False
                    other.state, other.message = 'error', link.message
                    other.metadata.update(error_code=code, raw_error=raw_error)
                    other.projection = None
                    other.dirty = True
                    self.changed.emit(other.session.id)
                self._transport = PowerShellLiveTransport()
        self.changed.emit(sid)
        # Coalesce bursts for one timer interval; don't loop synchronously from
        # each acknowledgement and starve the measurement event queue.


class LiveExcelControls(QWidget):
    def __init__(self, manager, session, parent=None, *, compact=False, status_label=None):
        super().__init__(parent)
        self.manager, self.session = manager, session
        self.compact = compact
        self._full_status = ''
        self._details = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0 if compact else 6)
        if compact:
            self.excel_button = QToolButton(self)
            self.excel_button.setText('Excel')
            self.excel_button.setAccessibleName('查看 Excel 与实时同步操作')
            self.excel_button.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
            indicator = self.excel_button.style().pixelMetric(QStyle.PixelMetric.PM_MenuButtonIndicator, None, self.excel_button)
            self.excel_button.setMinimumWidth(max(80, self.excel_button.fontMetrics().horizontalAdvance('Excel') + indicator + 28))
            menu = QMenu(self.excel_button)
            self.show_button = menu.addAction('查看 Excel')
            self.resync_button = menu.addAction('重新同步')
            self.stop_button = menu.addAction('停止同步')
            menu.addSeparator()
            self.details_action = menu.addAction('同步详情')
            self.details_action.triggered.connect(self._show_details)
            self.excel_button.setMenu(menu)
            # Keep the menu available even if the show action is disabled.
            self.excel_button.clicked.connect(lambda: manager.show(session.id))
            self.show_button.triggered.connect(lambda: manager.show(session.id))
            self.resync_button.triggered.connect(lambda: manager.resync(session))
            self.stop_button.triggered.connect(lambda: manager.stop(session.id))
            layout.addWidget(self.excel_button)
            self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        else:
            row = QHBoxLayout()
            self.show_button = QPushButton('查看 Excel')
            self.resync_button = QPushButton('重新同步')
            self.stop_button = QPushButton('停止同步')
            for button in (self.show_button, self.resync_button, self.stop_button):
                row.addWidget(button)
            row.addStretch(1)
            layout.addLayout(row)
            self.show_button.clicked.connect(lambda: manager.show(session.id))
            self.resync_button.clicked.connect(lambda: manager.resync(session))
            self.stop_button.clicked.connect(lambda: manager.stop(session.id))
        self.status_label = status_label if status_label is not None else QLabel(self)
        self.status_label.setWordWrap(not compact)
        if compact:
            self.status_label.setMinimumWidth(0)
            self.status_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            self.status_label.installEventFilter(self)
            if status_label is None:
                # Compact use normally supplies the existing page status row.
                # Without one, use the button tooltip/details without an extra row.
                self.status_label.hide()
        if status_label is None and not compact:
            layout.addWidget(self.status_label)
        manager.changed.connect(self._refresh)
        self._refresh(session.id)

    def eventFilter(self, watched, event):
        if watched is self.status_label and event.type() in (QEvent.Type.Resize, QEvent.Type.FontChange):
            self._render_status()
        return super().eventFilter(watched, event)

    def _render_status(self):
        if self.compact:
            single_line = ' · '.join(self._full_status.splitlines())
            text = self.status_label.fontMetrics().elidedText(single_line, Qt.TextElideMode.ElideRight,
                                                              max(0, self.status_label.contentsRect().width()))
            self.status_label.setText(text)
            self.excel_button.setToolTip(self._full_status + '\n主按钮：查看 Excel；下拉：同步操作与详情')
        else:
            self.status_label.setText(self._full_status)
        self.status_label.setToolTip(self._full_status)
        self.status_label.setAccessibleDescription(self._full_status)

    def _show_details(self):
        details = self._details if self._details is not None else QMessageBox(self)
        details.setWindowTitle('Excel 同步详情')
        details.setTextFormat(Qt.TextFormat.PlainText)
        details.setIcon(QMessageBox.Icon.Information)
        details.setText(self._full_status)
        status = self.manager.status(self.session.id)
        lines = ['已同步表示 Excel 已收到数据，文件仍由您在 Excel 中保存。']
        if status.get('path'):
            lines.append('文件位置：' + str(status['path']))
        if status.get('input_revision', -1) >= 0:
            lines.append('已确认输入修订：' + str(status['input_revision']))
        if status.get('pending'):
            lines.append('仍有最新数据等待同步。')
        details.setInformativeText('\n'.join(lines))
        details.setDetailedText('原始错误信息：\n' + str(status['raw_error']) if status.get('raw_error') else '')
        details.setStandardButtons(QMessageBox.StandardButton.Close)
        details.button(QMessageBox.StandardButton.Close).setText('关闭')
        # Reuse the parent-owned dialog. Deleting a QMessageBox with an expanded
        # details widget from its finished callback is unsafe on some Qt builds.
        self._details = details
        details.open()

    def _refresh(self, sid):
        if sid != self.session.id:
            return
        status = self.manager.status(sid)
        message = status['message']
        if status.get('connected'):
            message += ' · ' + ('文件已保存' if status.get('saved') else '文件尚未保存')
            if status.get('name'):
                message += ' · ' + status['name']
            if self.session.workspace_options.get('mode') == 'diameter':
                message += '\nExcel 模板均值按分组计算；FDM 面板显示原始均值，两者可能不同。'
        warnings = status.get('warnings') or []
        if warnings:
            message += '\n' + '\n'.join(warnings)
        self._full_status = message
        self._render_status()
        if self._details is not None:
            self._details.setText(message)
            self._details.setDetailedText('原始错误信息：\n' + str(status['raw_error']) if status.get('raw_error') else '')
        self.show_button.setEnabled(status.get('connected', False))
        self.stop_button.setEnabled(status.get('active', False) or status.get('connected', False))
        self.resync_button.setEnabled(status['state'] != 'detaching')
