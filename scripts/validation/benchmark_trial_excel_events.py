"""Measure desktop Excel from a real Qt event loop, using disposable workbooks.

No QTest wait loop or flush call drives the measurements. Source and profile
paths are explicit so the same probe can compare a clean baseline and a change.
The final acknowledgement is a software observation, not optical screen timing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--content-template', type=Path, required=True)
    parser.add_argument('--diameter-template', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--samples', type=int, default=100)
    parser.add_argument('--timeout', type=int, default=900)
    args = parser.parse_args()
    if sys.platform != 'win32':
        parser.error('Windows desktop Excel is required')
    if args.rounds < 1 or args.samples < 1:
        parser.error('Positive sample counts are required')
    source, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(source / 'src'))
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from fdm import settings
    from fdm.live_experiment import ExperimentSession, MaterialSnapshot
    from fdm.live_experiment.excel_template import save_template_preferences
    from fdm.live_experiment.excel_live import PowerShellLiveTransport
    from fdm.ui.live_excel import LiveExcelManager

    settings.settings_file_path = lambda: output / 'profile' / 'settings.json'
    clock = time.perf_counter
    report = dict(passed=False, source=str(source), platform=sys.platform,
                  samples=[], startup=[], operations=[], events=[], metrics={},
                  boundary='final input acknowledgement; not optical timing')
    templates = (args.content_template.resolve(), args.diameter_template.resolve())
    report['template_hashes'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in templates}
    report['source_hashes'] = {str(p): hashlib.sha256((source / p).read_bytes()).hexdigest()
        for p in ('src/fdm/ui/live_excel.py', 'src/fdm/live_experiment/excel_live.py',
                  'src/fdm/live_experiment/excel_live.ps1')}
    replies = {}

    class Transport(PowerShellLiveTransport):
        def request(self, operation, session_id='', generation=0, input_revision=0,
                    payload=None, **kwargs):
            started = clock()
            result = super().request(operation, session_id, generation, input_revision, payload, **kwargs)
            trace = dict(operation=operation, session_id=session_id, revision=input_revision,
                         request_start=started, request_return=clock(), response=result)
            report['operations'].append(trace)
            if operation in ('open', 'sync'):
                replies[(session_id, input_revision)] = trace
            return result

    app = QApplication([])
    manager = LiveExcelManager(transport=Transport())
    sessions = []
    active = baseline = pending = None
    mode_index, sample_index = -1, 0
    phase = 'between'
    started = clock()

    def persist():
        (output / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    def fail(exc):
        nonlocal phase
        if phase == 'failed':
            return
        phase = 'failed'
        report.update(error=str(exc), traceback=traceback.format_exc())
        persist()
        app.quit()

    def safe(callback):
        def invoke():
            try:
                callback()
            except Exception as exc:
                fail(exc)
        return invoke

    def metrics(values):
        ordered = sorted(values)
        return dict(n=len(values), p50_ms=statistics.median(values),
                    p95_ms=ordered[math.ceil(len(values) * .95) - 1], max_ms=max(values))

    @safe
    def finish():
        if not all(manager.detach(session) for session in sessions):
            QTimer.singleShot(30, finish)
            return
        if not manager.close():
            QTimer.singleShot(30, finish)
            return
        for mode in ('content', 'diameter'):
            for round_number in range(1, args.rounds + 1):
                rows = [s for s in report['samples'] if s['mode'] == mode and s['round'] == round_number]
                report['metrics'][f'{mode}-{round_number}'] = {
                    key: metrics([s[key] for s in rows]) for key in
                    ('input_to_final_ms', 'interaction_on_ms', 'interaction_off_ms', 'rpc_ms')}
        assert report['template_hashes'] == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in templates}
        report.update(passed=True, elapsed_ms=(clock() - started) * 1000)
        persist()
        app.quit()

    @safe
    def next_mode():
        nonlocal active, baseline, mode_index, sample_index, pending, phase
        mode_index += 1
        if mode_index >= 2:
            phase = 'detach'
            finish()
            return
        mode = ('content', 'diameter')[mode_index]
        save_template_preferences({'template': str(templates[mode_index])}, trial_mode=mode)
        active = ExperimentSession(workspace_options={'mode': mode}, save_images=False)
        active.add_operator('实时输出性能验证员')
        active.set_materials([MaterialSnapshot('one', '绵羊毛', 1.31), MaterialSnapshot('two', '山羊绒', 1.31)])
        active.configure(sample_name='独立性能验证', sample_number=mode)
        baseline = ExperimentSession.from_dict(active.to_dict())
        sessions.append(active)
        sample_index = 0
        phase = 'startup'
        pending = dict(t0=clock(), revision=active.revision)
        manager.ensure(active)

    @safe
    def add_sample():
        nonlocal sample_index, pending, phase
        sample_index += 1
        value = 20 + sample_index % 100 / 100
        # Alternate call order to reduce systematic warm-up/order bias.
        if sample_index % 2:
            before = clock()
            baseline.add_measurement(value, 'one')
            off_ms = (clock() - before) * 1000
        t0 = clock()
        active.add_measurement(value, 'one')
        manager.update(active)
        on_ms = (clock() - t0) * 1000
        if not sample_index % 2:
            before = clock()
            baseline.add_measurement(value, 'one')
            off_ms = (clock() - before) * 1000
        pending = dict(t0=t0, revision=active.revision, sample=sample_index,
                       round=(sample_index - 1) // args.samples + 1,
                       interaction_on_ms=on_ms, interaction_off_ms=off_ms)
        phase = 'sample'

    def changed(sid):
        nonlocal phase, pending
        try:
            state = manager.status(sid)
            report['events'].append(dict(time=clock(), session_id=sid, **state))
            if state['state'] in ('error', 'unsupported', 'stopped'):
                raise RuntimeError(state)
            if phase not in ('startup', 'sample') or active is None or sid != active.id or pending is None:
                return
            if state.get('input_revision') != pending['revision'] or state.get('pending'):
                return
            trace = replies.get((sid, pending['revision']))
            if trace is None or not trace['response'].get('ok'):
                return
            now = clock()
            row = {**pending, **trace, 'mode': ('content', 'diameter')[mode_index],
                   'final_ui': now, 'input_to_final_ms': (now - pending['t0']) * 1000,
                   'rpc_ms': (trace['request_return'] - trace['request_start']) * 1000}
            report['samples' if phase == 'sample' else 'startup'].append(row)
            pending = None
            phase = 'between'
            if sample_index % 20 == 0:
                persist()
            QTimer.singleShot(20, add_sample if sample_index < args.rounds * args.samples else next_mode)
        except Exception as exc:
            fail(exc)

    manager.changed.connect(changed)
    watchdog = QTimer()
    watchdog.setSingleShot(True)
    watchdog.timeout.connect(lambda: fail('Probe watchdog expired'))
    watchdog.start(args.timeout * 1000)
    QTimer.singleShot(0, next_mode)
    app.exec()
    print(json.dumps({'passed': report['passed'], 'samples': len(report['samples']),
                      'metrics': report['metrics']}, ensure_ascii=False))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
