"""Application-owned live Excel links; page widgets only display/control links."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from queue import Queue
import threading
import time
from uuid import uuid4

from PySide6.QtCore import QEvent, QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QMenu, QMessageBox, QPushButton,
                               QSizePolicy, QStyle, QToolButton, QVBoxLayout, QWidget)

from fdm.live_experiment.excel_template import (DiameterTemplateProfile, ExportSnapshot, TemplateProfile,
                                               template_preferences, working_template)
from fdm.live_experiment.excel_live import PowerShellLiveTransport, input_patch, project_inputs, report_update


_EXCEL_ERROR_MESSAGES = {
    'workbook_closed': '工作簿已关闭，重新同步可恢复。',
    'excel_busy': 'Excel 正忙，恢复后将继续同步最新数据。',
    'template_missing': '找不到 Excel 模板，请在报告页选择模板后重新同步。',
    'template_method': 'Excel 模板缺少当前试验方法，或存在重复匹配；请检查方法映射。',
    'template_mapping': 'Excel 模板缺少指定工作表或输入位置；请检查模板映射。',
    'template_macro': 'Excel 模板宏无法完成；测量数据与报表统计状态分别显示，请检查宏权限和模板。',
    'batch_pending': 'Excel 仍有未完成批次，将先恢复该批次后继续输出。',
    'batch_identity': 'Excel 批次身份与内容不一致，未确认写入；请查看同步详情。',
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


@dataclass(frozen=True)
class _Batch:
    operation: str
    generation: int
    revision: int
    projection: object
    payload: dict


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
    actions: list = field(default_factory=list)
    pending_batch: _Batch | None = None
    pending_auxiliary: _Batch | None = None
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
        self._busy_input_confirmed = True
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
            sid, generation, operation, revision, snapshot, previous, force_full, batch, options = task
            used_generation = generation
            projection = previous
            try:
                if operation == 'shutdown':
                    self._transport.close()
                    result = {'ok': True, 'status': 'closed'}
                elif batch is not None:
                    operation, used_generation, revision, projection = (
                        batch.operation, batch.generation, batch.revision, batch.projection)
                    result = self._transport.request(operation, sid, used_generation, revision, batch.payload,
                        progress=lambda value: self._emit_progress((value, batch, generation)))
                elif operation == 'recalculate' and previous is not None:
                    # The associated workbook retains its original mapping even
                    # if the template path was subsequently moved or reselected.
                    payload = {'profile': previous.data()['profile'], 'batch_id': uuid4().hex}
                    batch = _Batch(operation, generation, revision, previous, payload)
                    result = self._transport.request(operation, sid, generation, revision, payload)
                elif operation in ('open', 'sync', 'update_report', 'recalculate'):
                    mode = snapshot.trial_mode
                    preferences = template_preferences(trial_mode=mode)
                    if operation == 'update_report' and previous is not None:
                        data = previous.data()
                        profile_type = DiameterTemplateProfile if data['trial_kind'] == 'diameter' else TemplateProfile
                        profile = profile_type(**data['profile'])
                        template = data['template']
                    else:
                        profile = TemplateProfile.load(preferences.get('mapping'), trial_mode=mode)
                        template = working_template(create=True, trial_mode=mode)
                        if not template.is_file():
                            raise FileNotFoundError('试验 Excel 模板不存在，请在报告页选择模板后重新同步。')
                    method_labels = preferences.get('method_labels', {})
                    method_label = method_labels.get(snapshot.method) if isinstance(method_labels, dict) else None
                    projection = project_inputs(snapshot, template, profile, method_label=method_label)
                    if previous and operation in ('open', 'sync'):
                        old, current = previous.data(), projection.data()
                        if any(old[key] != current[key] for key in ('template', 'profile', 'trial_kind')):
                            operation = 'open'
                            used_generation += 1
                            force_full = True
                    if operation == 'update_report':
                        payload = report_update(projection, **options)
                    elif operation == 'recalculate':
                        payload = {'profile': projection.data()['profile']}
                    else:
                        payload = input_patch(None if force_full or operation == 'open' else previous, projection)
                        if operation == 'open':
                            payload.update(report_update(projection, information=True, settings=True))
                    if payload is None:
                        result = {'ok': True, 'status': 'synced', 'unchanged': True, 'input_written': True}
                    else:
                        payload['batch_id'] = uuid4().hex
                        batch = _Batch(operation, used_generation, revision, projection, payload)
                        result = self._transport.request(operation, sid, used_generation, revision, payload,
                            progress=lambda value: self._emit_progress((value, batch, generation)))
                else:
                    result = self._transport.request(operation, sid, generation, revision)
            except Exception as exc:
                result = {'ok': False, 'error_code': getattr(exc, 'code', 'excel_failed'),
                          'message': str(exc)}
            if self._destroyed_event.is_set():
                try:
                    self._transport.close()
                except Exception:
                    pass
                return
            try:
                self._finished.emit((sid, generation, operation, revision, projection, result, used_generation, batch))
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
            if link.state == 'error' and link.metadata.get('error_operation') in ('show', 'status', 'metadata', 'update_report', 'recalculate'):
                link.state, link.message = 'syncing', '正在等待测量数据输出'

    def status(self, session_id):
        link = self._links.get(self._sid(session_id))
        if not link:
            return {'state': 'idle', 'message': '进入试验后自动连接 Excel', 'active': False,
                    'connected': False, 'pending': False, 'unsynced': False}
        pending = bool((link.active and link.dirty) or link.pending_batch or link.pending_auxiliary or link.actions or link.detaching or
                       (self._busy and self._busy[0] == link.session.id))
        unsynced = bool(link.ever_connected and (link.dirty or link.projection is None or
                        link.session.revision != link.acknowledged_revision))
        input_pending = bool((link.active and link.dirty) or
            (link.pending_batch and link.pending_batch.operation in ('open', 'sync')) or
            (self._busy and self._busy[0] == link.session.id and self._busy[2] in ('open', 'sync')
             and not self._busy_input_confirmed))
        return {**link.metadata, 'state': link.state, 'message': link.message, 'active': link.active,
                'connected': link.connected, 'ever_connected': link.ever_connected, 'pending': pending, 'unsynced': unsynced,
                'input_pending': input_pending,
                'input_revision': link.acknowledged_revision, 'target_revision': link.session.revision,
                'generation': link.generation}

    def show(self, session_id):
        link = self._links.get(self._sid(session_id))
        if link and link.connected and not link.detaching:
            if not any(action[0] == 'show' for action in link.actions):
                link.actions.append(('show', None, {}))
            self._pump()

    def update_report(self, session, *, information=True, settings=False, snapshot=None):
        """Only an explicit user confirmation may overwrite Excel report fields."""
        if not information and not settings:
            return
        self._queue_report_action(session, 'update_report',
                                  {'information': bool(information), 'settings': bool(settings)}, snapshot=snapshot)

    def recalculate(self, session):
        # No FDM standard/group-width values are included in this command.
        self._queue_report_action(session, 'recalculate', {})

    def _queue_report_action(self, session, operation, options, *, snapshot=None):
        link = self._links.get(self._sid(session))
        if not link or not link.connected or link.detaching:
            return
        link.actions.append((operation, snapshot or ExportSnapshot.capture(link.session), options))
        self._pump()

    def retry(self, session):
        """Resume an uncertain fixed batch without rewriting report information."""
        link = self._links.get(self._sid(session))
        if not link or not link.connected:
            self.resync(session)
            return
        link.active = True
        link.retry_at = 0.0
        link.state, link.message = 'syncing', '正在重试测量数据输出'
        self.update(link.session)
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
        if link.state == 'error' and link.metadata.get('error_code') != 'excel_busy':
            link.pending_batch = None
        link.session = session
        link.active = True
        link.detaching = link.force_detach = False
        link.force_full = True
        link.dirty = True
        link.state, link.message = 'connecting', '正在重新输出测量数据；保留 Excel 报表信息'
        # Keep generation for the same known workbook; a manually closed book
        # gets a new generation. Never allow an old reply to confirm this one.
        if not link.connected and link.pending_batch is None and not (self._busy and self._busy[0] == sid):
            link.generation += 1
        link.retry_at = 0.0
        if link.connected and link.projection is None and link.pending_batch is None:
            # A workbook may exist although its first initialization failed.
            # Complete that initialization in the same book, not a quant-only
            # sync that would silently skip missing initial report settings.
            link.reopen = True
        # Busy/partly applied batches must finish unchanged before this rebuild.
        # Their cursor and absolute addresses make replay safe after an undo.
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
                 not (in_flight and not self._busy_input_confirmed))
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
        link.actions.clear()
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

    def _dispatch(self, sid, link, operation, snapshot=None, *, batch=None, options=None):
        generation = link.generation if link else 0
        revision = snapshot.revision if snapshot else (link.acknowledged_revision if link else 0)
        if batch is not None:
            generation, operation, revision = batch.generation, batch.operation, batch.revision
        self._busy = sid, generation, operation, revision
        self._busy_input_confirmed = operation not in ('open', 'sync')
        previous = (link.projection or link.workbook_projection) if link else None
        force_full = bool(link and link.force_full)
        if link and operation in ('open', 'sync') and batch is None:
            link.force_full = False
        self._tasks.put((sid, generation, operation, revision, snapshot, previous, force_full, batch, options or {}))

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
                    link.pending_batch = link.pending_auxiliary = None
                    link.state, link.message = 'detached', '同步已停止；Excel 工作簿由用户保存'
                    self.changed.emit(sid)
                    continue
                self._dispatch(sid, link, 'detach')
                return
            if (link.active and link.pending_auxiliary is not None and now >= link.retry_at
                    and link.state not in ('error', 'stopped', 'unsupported')):
                self._dispatch(sid, link, link.pending_auxiliary.operation, batch=link.pending_auxiliary)
                return
            if (link.active and link.pending_batch is not None and now >= link.retry_at
                    and link.state not in ('error', 'stopped', 'unsupported')):
                self._dispatch(sid, link, link.pending_batch.operation, batch=link.pending_batch)
                return
            if link.connected and link.actions and now >= link.retry_at:
                operation, snapshot, options = link.actions.pop(0)
                self._dispatch(sid, link, operation, snapshot, options=options)
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
        if self._closing:
            if now >= self._shutdown_retry and all(link.state == 'detached' for link in self._links.values()):
                self._dispatch('', None, 'shutdown')
            return
        for sid, link in self._links.items():
            if link.connected and link.active and now >= max(link.next_poll, link.retry_at):
                link.next_poll = now + 2.0
                self._dispatch(sid, link, 'status')
                return

    def _confirm_input(self, link, revision, projection):
        link.connected = link.ever_connected = True
        link.projection = link.workbook_projection = projection
        link.acknowledged_revision = revision
        link.metadata['inputs_written_revision'] = revision
        link.dirty = link.session.revision != revision or link.force_full
        if link.pending_batch is not None and link.pending_batch.revision == revision:
            link.pending_batch = None

    def _progress(self, value):
        event, batch, generation = value
        sid = event.get('session_id')
        link = self._links.get(sid)
        if (not link or not self._busy or self._busy[0] != sid
                or generation != link.generation
                or self._busy[2] not in ('open', 'sync') or self._busy[3] != batch.revision
                or event.get('generation') != batch.generation
                or event.get('input_revision') != batch.revision
                or event.get('batch_id') != batch.payload['batch_id']):
            return
        if event.get('event') == 'inputs_written' and event.get('input_written') is True:
            # This event is emitted after visible absolute writes, independently
            # of best-effort metadata or template calculation. It is an ACK.
            self._confirm_input(link, batch.revision, batch.projection)
            self._busy_input_confirmed = True
            link.metadata['last_input_batch'] = batch.payload['batch_id']
            link.metadata['write_elapsed_ms'] = event.get('elapsed_ms')
            link.state, link.message = 'synced', '测量数据已输出；报表由 Excel 计算'
            self.changed.emit(sid)

    @staticmethod
    def _metadata(link, result):
        if result.get('metadata_checked_at') and not result.get('metadata_error'):
            for key in ('metadata_error', 'metadata_error_code', 'metadata_error_stack', 'metadata_stage'):
                link.metadata.pop(key, None)
        for key in ('saved', 'save_state', 'has_file', 'path', 'name', 'excel_version',
                    'excel_pid', 'warnings', 'elapsed_ms', 'calculation_state',
                    'metadata_checked_at', 'last_input_at', 'timings', 'metadata_error',
                    'metadata_error_code', 'metadata_error_stack', 'metadata_stage', 'excel_created',
                    'batch_id', 'completed_ops', 'calculated_revision'):
            if key in result:
                link.metadata[key] = result[key]

    def _complete(self, value):
        sid, generation, operation, revision, projection, result, used_generation, batch = value
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
        self._metadata(link, result)
        quantitative = operation in ('open', 'sync')
        written = quantitative and (result.get('input_written') is True or
            (batch is not None and link.metadata.get('last_input_batch') == batch.payload['batch_id']
             and link.acknowledged_revision == revision))
        if written:
            self._confirm_input(link, revision, projection)
            if batch is not None:
                link.metadata['last_input_batch'] = batch.payload['batch_id']
            link.pending_batch = None
            if projection is not None:
                link.metadata['warnings'] = projection.data()['warnings']
        if result.get('ok'):
            if quantitative and not written:
                # Never promote an unconfirmed or unrelated successful reply to
                # the input baseline; this also catches malformed worker replies.
                result = {**result, 'ok': False, 'error_code': 'protocol_mismatch',
                          'message': 'Excel 未确认本批测量数据写入。'}
            else:
                if not result.get('unchanged'):
                    link.next_poll = time.monotonic() + 2.0
                if (batch is not None and link.pending_batch is not None
                        and link.pending_batch.payload['batch_id'] == batch.payload['batch_id']):
                    link.pending_batch = None
                if (batch is not None and link.pending_auxiliary is not None
                        and link.pending_auxiliary.payload['batch_id'] == batch.payload['batch_id']):
                    link.pending_auxiliary = None
                error_operation = link.metadata.get('error_operation')
                if quantitative or error_operation == operation or error_operation in ('status', 'show', 'metadata'):
                    for key in ('raw_error', 'error_code', 'error_stack', 'stage', 'error_operation'):
                        link.metadata.pop(key, None)
                if operation == 'detach':
                    link.connected = link.active = link.detaching = False
                    link.pending_batch = link.pending_auxiliary = None
                    link.state, link.message = 'detached', '同步已停止；Excel 工作簿由用户保存'
                elif quantitative:
                    link.state, link.message = 'synced', ('测量数据已输出；报表由 Excel 计算'
                        if not link.dirty else '正在等待下一批测量数据输出')
                elif operation in ('update_report', 'recalculate'):
                    if operation == 'recalculate' or (batch and batch.payload.get('report_settings') is not None):
                        for key in ('report_error', 'report_error_code', 'report_error_stack'):
                            link.metadata.pop(key, None)
                    if link.metadata.get('error_operation') in ('open', 'sync'):
                        link.state = 'error'
                    else:
                        link.state, link.message = 'synced', ('报表信息已更新' if operation == 'update_report' else '报表统计已请求')
                elif not link.detaching and not link.metadata.get('error_code'):
                    link.state, link.message = 'synced', '测量数据已输出；报表由 Excel 计算'
        if not result.get('ok'):
            code = result.get('error_code', result.get('code', 'excel_failed'))
            raw_error = result.get('message') or result.get('error') or ''
            link.metadata.update(error_code=code, raw_error=raw_error, error_operation=operation,
                                 save_state='unknown', saved=None)
            for key in ('stage', 'error_stack'):
                if key in result:
                    link.metadata[key] = result[key]
            link.state = 'unsupported' if code == 'unsupported_platform' else 'stopped' if code == 'workbook_closed' else 'error'
            link.message = _EXCEL_ERROR_MESSAGES.get(code, 'Excel 操作未完成，请查看同步详情。')
            if written:
                link.metadata['error_operation'] = 'metadata'
                if code not in ('workbook_closed', 'unsupported_platform'):
                    link.state, link.message = 'synced', '测量数据已输出；Excel 状态暂不可读取'
            elif operation in ('update_report', 'recalculate') and code != 'excel_busy':
                link.metadata.update(report_error=link.message, report_error_code=code,
                                     report_error_stack=result.get('error_stack', ''), calculation_state='unknown')
            if quantitative and not written:
                link.dirty = True
            if result.get('name') or result.get('workbook_connected'):
                link.connected = link.ever_connected = True
            # Keep precisely the batch that may already have applied a prefix.
            # Diffing a newer state against the old ACK can otherwise miss undo.
            if batch is not None and not written and quantitative:
                link.pending_batch = batch
            elif batch is not None and operation in ('update_report', 'recalculate'):
                link.pending_auxiliary = batch if code == 'excel_busy' else None
            if code == 'excel_busy':
                link.state, link.message = 'busy', 'Excel 正在编辑、保存或运行宏；恢复后继续输出'
                link.retry_at = time.monotonic() + .5
                if operation == 'show':
                    link.actions.insert(0, ('show', None, {}))
                # Readonly busy never sets dirty, invalidates the projection or
                # expands the next write into a report-wide synchronization.
            elif operation == 'detach':
                link.detaching = False
            if code in ('workbook_closed', 'worker_timeout', 'worker_exited', 'worker_disconnected',
                        'protocol_mismatch', 'unsupported_platform'):
                link.connected = link.active = False
                link.pending_batch = link.pending_auxiliary = None
            if code in ('worker_timeout', 'worker_exited', 'worker_disconnected', 'protocol_mismatch'):
                for other in self._links.values():
                    other.connected = other.active = False
                    other.state, other.message = 'error', link.message
                    other.metadata.update(error_code=code, raw_error=raw_error, save_state='unknown', saved=None)
                    other.pending_batch = other.pending_auxiliary = None
                    other.dirty = other.dirty or other.session.revision != other.acknowledged_revision
                    self.changed.emit(other.session.id)
                self._transport = PowerShellLiveTransport()
        self.changed.emit(sid)
        # Let Qt return to measurement interaction before dispatching the next
        # coalesced request. A metadata/auxiliary failure is not an input failure.


class ReportInformationDialog(QDialog):
    """Review explicit Excel-owned fields before overwriting any of them."""
    def __init__(self, session, parent=None):
        super().__init__(parent)
        from fdm.live_experiment.statistics import resolve_method
        self.snapshot = ExportSnapshot.capture(session)
        self.setWindowTitle('更新报表信息')
        self.setMinimumWidth(440)
        layout = QVBoxLayout(self)
        note = QLabel('仅更新勾选的项目，将覆盖 Excel 中对应的信息；未勾选的内容保持不变。', self)
        note.setWordWrap(True)
        layout.addWidget(note)
        self.information = QCheckBox('样品／人员信息', self)
        self.information.setChecked(True)
        layout.addWidget(self.information)
        names = '、'.join(operator.name for operator in session.operators.values()) or '（空）'
        date = datetime.fromisoformat(session.created_at).astimezone().date().isoformat()
        info = QLabel(f'样品编号：{session.sample_number or "（空）"}\n样品名称：{session.sample_name or "（空）"}'
                      f'\n试验日期：{date}\n操作员：{names}', self)
        info.setTextFormat(Qt.TextFormat.PlainText)
        info.setWordWrap(True)
        layout.addWidget(info)
        self.settings = QCheckBox('标准／组距（更新后重新统计一次）', self)
        layout.addWidget(self.settings)
        settings = QLabel(f'FDM 标准：{resolve_method(session.method).label}\n组距：{session.group_width:g} μm'
                          f'\n分组：{"启用" if session.grouped else "未启用"}', self)
        settings.setWordWrap(True)
        layout.addWidget(settings)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        self.apply_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.apply_button.setText('更新选中信息')
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText('取消')
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        for check in (self.information, self.settings):
            check.toggled.connect(lambda: self.apply_button.setEnabled(self.information.isChecked() or self.settings.isChecked()))
        layout.addWidget(buttons)


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
            self.resync_button = menu.addAction('重新输出测量数据')
            self.update_report_action = menu.addAction('更新报表信息…')
            self.recalculate_action = menu.addAction('重新统计报表')
            self.stop_button = menu.addAction('停止同步')
            menu.addSeparator()
            self.details_action = menu.addAction('同步详情')
            self.details_action.triggered.connect(self._show_details)
            self.excel_button.setMenu(menu)
            # Keep the menu available even if the show action is disabled.
            self.excel_button.clicked.connect(lambda: manager.show(session.id))
            self.show_button.triggered.connect(lambda: manager.show(session.id))
            self.resync_button.triggered.connect(lambda: manager.resync(session))
            self.update_report_action.triggered.connect(self._update_report)
            self.recalculate_action.triggered.connect(lambda: manager.recalculate(session))
            self.stop_button.triggered.connect(lambda: manager.stop(session.id))
            layout.addWidget(self.excel_button)
            self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        else:
            row = QHBoxLayout()
            self.show_button = QPushButton('查看 Excel')
            self.resync_button = QPushButton('重新输出测量数据')
            self.stop_button = QPushButton('停止同步')
            self.update_report_action = QPushButton('更新报表信息…')
            self.recalculate_action = QPushButton('重新统计报表')
            for button in (self.show_button, self.resync_button, self.update_report_action, self.recalculate_action, self.stop_button):
                row.addWidget(button)
            row.addStretch(1)
            layout.addLayout(row)
            self.show_button.clicked.connect(lambda: manager.show(session.id))
            self.resync_button.clicked.connect(lambda: manager.resync(session))
            self.update_report_action.clicked.connect(self._update_report)
            self.recalculate_action.clicked.connect(lambda: manager.recalculate(session))
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

    def _update_report(self):
        dialog = ReportInformationDialog(self.session, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.manager.update_report(self.session, information=dialog.information.isChecked(),
                                       settings=dialog.settings.isChecked(), snapshot=dialog.snapshot)
        dialog.deleteLater()

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
        lines = ['测量数据已输出只表示原始输入已写入；报表计算及文件保存分别显示。',
                 '重新输出只重建测量数据；样品、人员、标准及组距由 Excel 维护，需明确选择更新。']
        if status.get('path'):
            lines.append('文件位置：' + str(status['path']))
        if status.get('input_revision', -1) >= 0:
            lines.append('已确认输入修订：' + str(status['input_revision']))
        if status.get('input_pending'):
            lines.append('仍有最新数据等待同步。')
        details.setInformativeText('\n'.join(lines))
        details.setDetailedText(self._diagnostics(status))
        details.setStandardButtons(QMessageBox.StandardButton.Close)
        details.button(QMessageBox.StandardButton.Close).setText('关闭')
        # Reuse the parent-owned dialog. Deleting a QMessageBox with an expanded
        # details widget from its finished callback is unsafe on some Qt builds.
        self._details = details
        details.open()

    @staticmethod
    def _diagnostics(status):
        labels = (('raw_error', '原始错误'), ('metadata_error', '状态读取错误'),
                  ('report_error', '报表操作错误'), ('report_error_stack', '报表操作位置'),
                  ('metadata_stage', '状态读取阶段'), ('metadata_error_stack', '状态读取位置'),
                  ('stage', '操作阶段'), ('error_stack', '调用位置'),
                  ('metadata_checked_at', '状态检查时间'), ('last_input_at', '最近输入时间'))
        return '\n'.join(label + '：' + str(status[key]) for key, label in labels if status.get(key))

    def _refresh(self, sid):
        if sid != self.session.id:
            return
        status = self.manager.status(sid)
        message = status['message']
        if status.get('connected'):
            save_state = status.get('save_state', 'unknown')
            message += ' · ' + {'never_saved': '尚未另存文件', 'unsaved_changes': '文件有未保存修改',
                                'saved': '文件已保存', 'unknown': '文件保存状态暂不可知'}.get(save_state, '文件保存状态暂不可知')
            calculation = status.get('calculation_state')
            if calculation in ('manual', 'calculating', 'pending', 'unknown'):
                message += ' · ' + {'manual': 'Excel 手动计算，报表可能尚未更新',
                                    'calculating': 'Excel 正在计算报表', 'pending': '报表等待计算',
                                    'unknown': '报表计算状态暂不可知'}[calculation]
            if status.get('name'):
                message += ' · ' + status['name']
            if self.session.workspace_options.get('mode') == 'diameter':
                message += '\nExcel 模板均值按分组计算；FDM 面板显示原始均值，两者可能不同。'
        warnings = status.get('warnings') or []
        if status.get('report_error'):
            message += '\n报表操作未完成：' + status['report_error']
        if warnings:
            message += '\n' + '\n'.join(warnings)
        self._full_status = message
        self._render_status()
        if self._details is not None:
            self._details.setText(message)
            self._details.setDetailedText(self._diagnostics(status))
        self.show_button.setEnabled(status.get('connected', False))
        self.stop_button.setEnabled(status.get('active', False) or status.get('connected', False))
        self.resync_button.setEnabled(status['state'] != 'detaching')
        for action in (self.update_report_action, self.recalculate_action):
            action.setEnabled(status.get('connected', False) and status['state'] != 'detaching')
