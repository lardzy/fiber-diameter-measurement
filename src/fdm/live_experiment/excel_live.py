"""Immutable Excel input projection and one long-lived STA worker transport.

No Qt, COM or image evidence is part of this contract. A projection is confirmed
only by the matching final acknowledgement, never by the input-written event.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
from queue import Empty, Queue
import re
import subprocess
import sys
import threading
import time
from uuid import uuid4

from .excel_template import (DiameterTemplateProfile, ExcelTemplateError,
                             column_name, template_request)


_CELL = re.compile(r"^([A-Z]+)([1-9][0-9]*)$")


def _cell(value):
    match = _CELL.fullmatch(value)
    if not match:
        raise ValueError(f"模板单元格无效：{value}")
    column = 0
    for char in match[1]:
        column = column * 26 + ord(char) - 64
    return column, int(match[2])


@dataclass(frozen=True)
class InputProjection:
    """Canonical JSON is immutable and excludes evidence/session revisions."""
    encoded: str

    def data(self):
        return json.loads(self.encoded)


def project_inputs(snapshot, template, profile, *, method_label=None):
    request = template_request(snapshot, template, Path(template).with_suffix('.unused.xls'), profile, method_label=method_label)
    sheet, first = profile.data_sheet, profile.first_data_row
    columns = (profile.columns if isinstance(profile, DiameterTemplateProfile) else
               tuple(start + offset for pair in profile.column_starts for start in pair for offset in range(9)))
    metadata = {}
    # Enumerate only the small metadata rows, not the raw-data clear ranges.
    for address in request['clears']:
        start, _, end = address.partition(':')
        c1, r1 = _cell(start)
        c2, r2 = _cell(end or start)
        if r1 >= first:
            continue
        if r2 - r1 > 100 or c2 - c1 > 256:
            raise ValueError('模板元数据映射范围过大')
        for column in range(c1, c2 + 1):
            for row in range(r1, r2 + 1):
                metadata[f'{column_name(column)}{row}'] = None
    for item in request['cells']:
        metadata[item['cell']] = item['value']
    raw = {column_name(column): [] for column in columns}
    for block in request['blocks']:
        column, row = _cell(block['cell'])
        if row != first:
            raise ValueError('原始数据首行与映射不一致')
        raw[column_name(column)] = [row[0] for row in block['values']]
    managed = [{'sheet': sheet, 'range': f'{column}{first}:{column}65536'} for column in raw]
    managed += [{'sheet': sheet, 'range': cell} for cell in metadata]
    control_sheet = getattr(profile, 'group_width_sheet', getattr(profile, 'standard_sheet', sheet))
    managed.append({'sheet': control_sheet, 'range': profile.group_width_cell})
    if not isinstance(profile, DiameterTemplateProfile):
        managed.append({'sheet': profile.standard_sheet, 'range': profile.method_index_cell})
    data = {'template': request['template'], 'trial_kind': request['trial_kind'],
            'profile': request['profile'], 'metadata': metadata, 'raw': raw,
            'checks': request['checks'], 'warnings': request['warnings'],
            'group_width': request['group_width'], 'grouped': request['grouped'],
            'method_label': request.get('method_label'), 'managed_ranges': managed}
    return InputProjection(json.dumps(data, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(',', ':')))


def input_patch(previous: InputProjection | None, current: InputProjection):
    """Append only new values; changed/deleted columns rewrite and clear tails."""
    now = current.data()
    old = previous.data() if previous else None
    reset = old is None or any(old[key] != now[key] for key in ('template', 'trial_kind', 'profile'))
    if not reset and previous == current:
        return None
    sheet = now['profile']['data_sheet']
    first = now['profile']['first_data_row']
    payload = {key: now[key] for key in ('template', 'trial_kind', 'profile', 'checks', 'warnings',
                                         'group_width', 'grouped', 'method_label', 'managed_ranges')}
    cells, blocks, clears = [], [], []
    for address, value in now['metadata'].items():
        if reset or old['metadata'].get(address) != value:
            if value is None:
                clears.append({'sheet': sheet, 'range': address})
            else:
                cells.append({'sheet': sheet, 'cell': address, 'value': value})
    if old and not reset:
        for address in old['metadata'].keys() - now['metadata'].keys():
            clears.append({'sheet': sheet, 'range': address})
    for column, values in now['raw'].items():
        before = [] if reset else old['raw'].get(column, [])
        if reset:
            clears.append({'sheet': sheet, 'range': f'{column}{first}:{column}65536'})
        if not reset and before == values:
            continue
        append = not reset and len(values) >= len(before) and values[:len(before)] == before
        offset = len(before) if append else 0
        if values[offset:]:
            blocks.append({'sheet': sheet, 'cell': f'{column}{first + offset}',
                           'values': [[value] for value in values[offset:]]})
        if not reset and len(before) > len(values):
            clears.append({'sheet': sheet, 'range': f'{column}{first + len(values)}:{column}{first + len(before) - 1}'})
    payload.update(cells=cells, blocks=blocks, clears=clears, full_sync=reset)
    return payload


class PowerShellLiveTransport:
    """Serial JSON-lines RPC. Killing this *worker* never kills the user's Excel."""
    def __init__(self, *, timeout=90.0):
        self.timeout = timeout
        self._process = None
        self._lines = Queue()
        self._stderr = deque(maxlen=40)
        self._lock = threading.Lock()
        self._closed = False

    def _start(self):
        if self._closed:
            raise ExcelTemplateError('Excel 同步组件已关闭', code='worker_closed')
        if sys.platform != 'win32':
            raise ExcelTemplateError('Excel 实时同步需要 Windows 桌面版 Microsoft Excel；当前仍可继续测量。', code='unsupported_platform')
        if self._process:
            if self._process.poll() is not None:
                raise ExcelTemplateError('Excel 同步组件已退出，请重新同步。', code='worker_exited')
            return
        script = Path(__file__).with_suffix('.ps1')
        command = ['powershell.exe', '-NoLogo', '-NoProfile', '-NonInteractive', '-STA',
                   '-ExecutionPolicy', 'Bypass', '-File', str(script)]
        try:
            self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace', bufsize=1,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except OSError as exc:
            raise ExcelTemplateError(f'无法启动 Excel 同步组件：{exc}', code='worker_start_failed') from exc
        def read_stdout():
            for line in self._process.stdout:
                self._lines.put(line)
            self._lines.put(None)
        def read_stderr():
            for line in self._process.stderr:
                self._stderr.append(line.rstrip())
        threading.Thread(target=read_stdout, daemon=True, name='excel-live-output').start()
        threading.Thread(target=read_stderr, daemon=True, name='excel-live-errors').start()

    def request(self, operation, session_id='', generation=0, input_revision=0, payload=None, *, progress=None):
        with self._lock:
            self._start()
            request = dict(operation=operation, request_id=uuid4().hex, session_id=session_id,
                           generation=generation, input_revision=input_revision, payload=payload or {})
            try:
                self._process.stdin.write(json.dumps(request, ensure_ascii=False, allow_nan=False) + '\n')
                self._process.stdin.flush()
            except (OSError, ValueError) as exc:
                raise ExcelTemplateError('Excel 同步连接已断开，请重新同步。', code='worker_disconnected') from exc
            deadline = time.monotonic() + self.timeout
            while True:
                try:
                    line = self._lines.get(timeout=max(0.001, deadline - time.monotonic()))
                except Empty as exc:
                    # A timed-out command can have committed part of its input.
                    # Never continue this stream or treat its late reply as ACK.
                    self.abort()
                    raise ExcelTemplateError('Excel 长时间无响应，最后一批数据状态未知；请检查 Excel 后重新同步。', code='worker_timeout') from exc
                if line is None:
                    raise ExcelTemplateError('Excel 同步组件已退出：' + '\n'.join(self._stderr)[-1200:], code='worker_exited')
                try:
                    reply = json.loads(line.lstrip('\ufeff'))
                except ValueError:
                    self._stderr.append(line.rstrip())
                    if time.monotonic() >= deadline:
                        self.abort()
                        raise ExcelTemplateError('Excel 同步协议无响应。', code='worker_timeout')
                    continue
                if not isinstance(reply, dict) or any(reply.get(key) != request[key] for key in
                        ('request_id', 'session_id', 'generation', 'input_revision')):
                    self.abort()
                    raise ExcelTemplateError('Excel 返回的试验身份或修订不匹配，未确认写入。', code='protocol_mismatch')
                if reply.get('event'):
                    if progress:
                        progress(reply)
                    continue
                return reply

    def abort(self):
        self._closed = True
        process = self._process
        if process and process.poll() is None:
            process.terminate()
        # Excel is intentionally never terminated: all workbooks belong to user.

    def close(self):
        if self._process and self._process.poll() is None and not self._closed:
            result = self.request('shutdown')
            if not result.get('ok'):
                raise ExcelTemplateError(result.get('message') or '无法释放 Excel 输入区，请关闭 Excel 中的模态窗口后重试。',
                                         code=result.get('error_code', 'detach_failed'))
        self._closed = True
        if self._process and self._process.stdin:
            self._process.stdin.close()
