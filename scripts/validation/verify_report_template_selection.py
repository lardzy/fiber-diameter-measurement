"""Report picker -> real Qt coordinator -> desktop Excel source identity probe.

Runs only in Windows, with an isolated profile and disposable template copies.
The file chooser result is injected; the actual report-panel selection method,
coordinator and production PowerShell transport are exercised. No user book is
selected or closed. Template originals are only read and hash-checked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-template', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != 'win32':
        parser.error('Windows desktop Excel required')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = args.source_template.resolve()
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / 'src'))
    from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
    from PySide6.QtTest import QTest
    from fdm import settings
    from fdm.live_experiment import ExperimentSession, MaterialSnapshot
    from fdm.live_experiment.excel_template import save_template_preferences
    from fdm.live_experiment.excel_live import PowerShellLiveTransport
    from fdm.ui import trial_report_panel
    from fdm.ui.live_excel import LiveExcelManager

    # Reuse the independent-PID Excel window accessor used by other probes.
    accessor = (repo / 'scripts/validation/verify_live_excel_worker.ps1').read_text(encoding='utf-8-sig')
    accessor = accessor.split("Add-Type -TypeDefinition @'", 1)[1].split("'@", 1)[0]
    probe = output / 'inspect-owned-book.ps1'
    probe.write_text("param([string]$RequestPath)\n$ErrorActionPreference='Stop'\nAdd-Type -TypeDefinition @'" + accessor + "'@\n" + r'''
$r=Get-Content -LiteralPath $RequestPath -Raw -Encoding UTF8|ConvertFrom-Json
$excel=$null;$book=$null
if($r.operation -eq 'prepare') {
    $before=@(Get-Process EXCEL -ErrorAction SilentlyContinue|ForEach-Object {$_.Id})
    Copy-Item -LiteralPath $r.source -Destination $r.destination -Force
    $excel=New-Object -ComObject Excel.Application
    Add-Type -TypeDefinition 'using System;using System.Runtime.InteropServices;public class ExcelPid { [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint p); }'
    [uint32]$pidValue=0;[void][ExcelPid]::GetWindowThreadProcessId([IntPtr]$excel.Hwnd,[ref]$pidValue)
    if($before -contains [int]$pidValue){throw 'Preparation did not create an independent Excel PID'}
    @{id=$pidValue;created=(Get-Process -Id $pidValue).StartTime.ToUniversalTime().Ticks.ToString()}|
        ConvertTo-Json|Set-Content -LiteralPath ($RequestPath+'.owner.json') -Encoding UTF8
    try {
        $excel.DisplayAlerts=$false
        $excel.EnableEvents=$false
        $book=$excel.Workbooks.Add([string]$r.source)
        $sheet=$book.Worksheets.Add();$sheet.Name='FDM template identity'
        $sheet.Range('A1').Value2=[string]$r.marker
        $book.CheckCompatibility=$false
        $book.SaveAs([string]$r.destination,17);$book.Close($false);$book=$null
        $result=@{marker=$r.marker;hash=(Get-FileHash -LiteralPath $r.destination).Hash}
    } finally { if($book){$book.Close($false)}; $excel.Quit() }
} else {
    $process=Get-Process -Id ([int]$r.excel_pid)
    if($process.ProcessName -ne 'EXCEL' -or $process.StartTime.ToUniversalTime().Ticks.ToString() -ne [string]$r.excel_created){throw 'Excel identity mismatch'}
    $window=[FdmExcelProbe]::Window([uint32]$r.excel_pid)
    if(!$window){throw 'Owned Excel window unavailable'}
    $excel=$window.Application
    if($r.operation -eq 'cleanup') {
        foreach($name in $r.names){$excel.Workbooks.Item([string]$name).Close($false)}
        $remaining=$excel.Workbooks.Count
        if($remaining -eq 0){$excel.Quit()}
        $result=@{remaining=$remaining}
    } else {
        $book=$excel.Workbooks.Item([string]$r.name)
        $sheet=$book.Worksheets.Item('FDM template identity')
        if($r.operation -eq 'note'){$sheet.Range('A2').Value2='manual note retained'}
        $data=$book.Worksheets.Item('原始数据')
        $result=@{name=$book.Name;marker=$sheet.Range('A1').Value2;note=$sheet.Range('A2').Value2;
            first=$data.Range('D25').Value2;second=$data.Range('D26').Value2;count=$excel.Workbooks.Count}
    }
}
$result|ConvertTo-Json -Depth 10|Set-Content -LiteralPath $r.result -Encoding UTF8
''', encoding='utf-8-sig')
    report = {'passed': False, 'source_sha256': source_hash, 'checks': [], 'requests': [], 'books': []}
    sequence = 0

    def inspect(operation, **values):
        nonlocal sequence
        sequence += 1
        request = output / f'request-{sequence}.json'
        result = output / f'result-{sequence}.json'
        request.write_text(json.dumps(dict(operation=operation, result=str(result), **values)), encoding='utf-8')
        completed = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-STA', '-ExecutionPolicy', 'Bypass',
                        '-File', str(probe), '-RequestPath', str(request)], capture_output=True, timeout=90)
        if completed.returncode:
            (output / f'probe-{sequence}.stderr.txt').write_bytes(completed.stderr)
            raise RuntimeError(completed.stderr.decode('utf-8', errors='replace'))
        return json.loads(result.read_text(encoding='utf-8-sig'))

    def check(condition, message):
        if not condition:
            raise AssertionError(message)
        report['checks'].append(message)

    settings.settings_file_path = lambda: output / 'profile' / 'settings.json'
    settings.settings_directory = lambda: output / 'profile'
    original = output / 'original.xlt'
    selected = output / 'selected.xlt'
    inspect('prepare', source=str(source), destination=str(original), marker='original-copy')
    inspect('prepare', source=str(source), destination=str(selected), marker='selected-v1')
    first_hash = hashlib.sha256(selected.read_bytes()).hexdigest()
    save_template_preferences({'template': str(original)})

    class Transport(PowerShellLiveTransport):
        def request(self, operation, session_id='', generation=0, input_revision=0, payload=None, **kwargs):
            result = super().request(operation, session_id, generation, input_revision, payload, **kwargs)
            report['requests'].append({'operation': operation, 'session': session_id, 'generation': generation,
                                       'template': (payload or {}).get('template'), 'result': result})
            return result

    app = QApplication.instance() or QApplication([])
    manager = LiveExcelManager(transport=Transport())
    windows = []
    books = {}

    def make():
        window = QMainWindow()
        session = window.session = ExperimentSession(save_images=False)
        session.add_operator('模板验证员')
        session.set_materials([MaterialSnapshot('a', '绵羊毛', 1.31)])
        session.add_measurement(18.25, 'a')
        central = QWidget(window)
        window.setCentralWidget(central)
        window.report_layout = QVBoxLayout(central)
        window._guard = lambda fn: fn()
        panel = trial_report_panel.TrialReportPanel(window)
        panel.bind_live_excel(manager)
        # Standard inspection is a separate read-only job; it is not the live
        # workbook selection and is already covered by the native worker probe.
        panel.inspect = lambda: None
        windows.append((window, panel))
        return session, panel

    def wait(session, generation):
        until = time.monotonic() + 90
        while time.monotonic() < until:
            QTest.qWait(10)
            state = manager.status(session.id)
            if state['state'] in ('error', 'unsupported', 'stopped'):
                raise AssertionError(state)
            if state['state'] == 'synced' and state['generation'] == generation and not state['pending']:
                books[state['name']] = state
                return state
        raise AssertionError(manager.status(session.id))

    def read(state, operation='read'):
        value = inspect(operation, excel_pid=state['excel_pid'], excel_created=state['excel_created'], name=state['name'])
        report['books'].append(value)
        return value

    try:
        first, panel = make()
        manager.ensure(first)
        old = wait(first, 1)
        check(read(old, 'note')['marker'] == 'original-copy', 'Initial workbook uses original template content')
        other, _ = make()
        manager.ensure(other)
        other_state = wait(other, 1)
        trial_report_panel.QFileDialog.getOpenFileName = lambda *a: (str(selected), '')
        panel.choose()
        new = wait(first, 2)
        check(read(new)['marker'] == 'selected-v1', 'Actual report selection opens chosen template contents')
        check(read(new)['first'] == 18.25, 'Existing measurement copied into replacement workbook')
        check(read(old)['note'] == 'manual note retained', 'Old workbook stays open with manual notes')
        check(str(selected) in panel.live_template_label.text(), 'Panel reports actual workbook source')
        check(first_hash == hashlib.sha256(selected.read_bytes()).hexdigest(), 'Live opening does not save template')
        inspect('prepare', source=str(source), destination=str(selected), marker='selected-v2')
        final_hash = hashlib.sha256(selected.read_bytes()).hexdigest()
        panel.choose()
        newest = wait(first, 3)
        check(read(newest)['marker'] == 'selected-v2', 'Same-path reselect loads replaced file contents')
        check(read(new)['marker'] == 'selected-v1', 'Previous selected workbook retained independently')
        other.add_measurement(22.5, 'a')
        manager.resync(other)
        other_state = wait(other, 1)
        value = read(other_state)
        check(value['marker'] == 'original-copy' and value['second'] == 22.5,
              'Other existing trial resync keeps its original workbook and template')
        future, _ = make()
        manager.ensure(future)
        check(read(wait(future, 1))['marker'] == 'selected-v2', 'Future trial adopts selected default')
        check(hashlib.sha256(selected.read_bytes()).hexdigest() == final_hash, 'Selected template unchanged by live writes')
        check(hashlib.sha256(source.read_bytes()).hexdigest() == source_hash, 'User source template unchanged')
        check(read(old)['note'] == 'manual note retained', 'Original manual note survives all replacement and synchronization')
        report['passed'] = True
    except Exception as exc:
        report['error'] = str(exc)
        raise
    finally:
        for window, panel in windows:
            panel.close()
            manager.stop(window.session)
        until = time.monotonic() + 30
        while not manager.close() and time.monotonic() < until:
            QTest.qWait(10)
        try:
            # Only the independently created manager instance and its recorded
            # disposable workbook names are eligible for probe cleanup.
            if books and manager.close():
                identity = next(iter(books.values()))
                report['cleanup'] = inspect('cleanup', excel_pid=identity['excel_pid'],
                    excel_created=identity['excel_created'], names=list(books))
        except Exception as exc:
            report['cleanup_error'] = str(exc)
        (output / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'passed': report['passed'], 'checks': report['checks']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
