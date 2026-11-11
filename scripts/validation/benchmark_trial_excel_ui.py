"""Paired full-trial UI timing with native Excel and a deterministic fake camera.

The active case runs the real controller, workspace, records dock and Excel
manager in QApplication.exec(). The disabled case uses the manager's real Stop
operation. Each pair has the same starting record count and alternates order.
No sleep, forced summary refresh, repaint, processEvents or flush drives samples.

Paint completion is a Qt software boundary, not compositor/monitor scan-out.
The existing 80 ms statistics timer is included ONLY in the end-to-end metric;
callback execution and paint execution are reported separately. --ui-only is a
local instrumentation smoke test with BOTH links stopped, never Excel evidence.
Native workbooks remain open after detach; identities are emitted for an owner
who wishes to close only the disposable workbooks/process created by this probe.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--content-template', type=Path)
    parser.add_argument('--diameter-template', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=100, help='paired samples per trial mode')
    parser.add_argument('--warmup', type=int, default=5, help='unscored pairs per trial mode')
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--ui-only', action='store_true')
    parser.add_argument('--scale', type=float, default=1.0, help='Process-only target DPR; system DPI is unchanged')
    args = parser.parse_args()
    if not args.ui_only and (sys.platform != 'win32' or not args.content_template or not args.diameter_template):
        parser.error('Native comparison requires Windows and both template paths')
    if args.samples < 1 or args.warmup < 0 or not math.isfinite(args.scale) or args.scale <= 0:
        parser.error('Invalid sample count')
    source, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('QT_QPA_PLATFORM', 'windows' if sys.platform == 'win32' else 'offscreen')
    for key in ('QT_SCALE_FACTOR', 'QT_SCREEN_SCALE_FACTORS', 'QT_FONT_DPI'):
        os.environ.pop(key, None)
    native_dpr = float(subprocess.check_output([sys.executable, '-c',
        'from PySide6.QtWidgets import QApplication; a=QApplication([]); print(a.primaryScreen().devicePixelRatio())'], text=True).strip())
    os.environ['QT_SCALE_FACTOR'] = str(args.scale / native_dpr)
    sys.path[:0] = [str(source / 'src'), str(source / 'scripts' / 'validation')]
    from PySide6.QtCore import QEvent, QTimer
    from PySide6.QtWidgets import QApplication
    from fdm.live_experiment import MaterialSnapshot
    from fdm.live_experiment.controller import CalibrationSnapshot, LiveExperimentController
    from fdm.live_experiment.excel_template import save_template_preferences
    from fdm.ui.live_experiment_workspace import LiveExperimentWorkspace, open_live_experiment
    from validate_workflow_redesign import WorkflowHarness, synthetic_frame

    clock = time.perf_counter
    report = dict(passed=False, native_excel=not args.ui_only, samples=[], startup=[], metrics={},
                  errors=[], platform=sys.platform, source=str(source), operations=[], workbooks={},
                  boundary='controller finish to Qt paint completion; not optical timing',
                  cpu_boundary='controller acceptance plus queued records/summary callbacks; no idle waiting',
                  sample_order='paired equal record counts, alternate off/on order per pair',
                  input='controller.start/finish(method=free), synthetic camera, real production UI slots')
    paths = ('src/fdm/ui/live_excel.py', 'src/fdm/ui/live_experiment_workspace.py',
             'src/fdm/ui/trial_records.py', 'src/fdm/live_experiment/controller.py',
             'src/fdm/live_experiment/excel_live.py', 'src/fdm/live_experiment/excel_live.ps1')
    report['source_hashes'] = {p: hashlib.sha256((source / p).read_bytes()).hexdigest() for p in paths}
    templates = dict(content=args.content_template, diameter=args.diameter_template)
    report['template_hashes'] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in templates.values() if p}
    current = None
    cases = []
    pending = None
    done = False
    phase = 'startup'
    mode_index = 0
    pair_number = 0
    pair_order = []
    pair_position = 0
    started = clock()

    def persist():
        (output / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    def fail(exc):
        nonlocal done
        if done:
            return
        done = True
        report['errors'].append(dict(error=str(exc), traceback=traceback.format_exc(), phase=phase,
                                     pending={k: v for k, v in (pending or {}).items() if k != 'paint_targets'}))
        persist()
        app.exit(1)

    def safe(fn):
        def call(*a, **kw):
            try:
                return fn(*a, **kw)
            except Exception as exc:
                fail(exc)
        return call

    class App(QApplication):
        def notify(self, receiver, event):
            sample = pending
            target = sample and sample.get('paint_targets', {}).get(id(receiver))
            if target and event.type() == QEvent.Type.Paint:
                before = clock()
                value = super().notify(receiver, event)
                after = clock()
                # The callback runs AFTER Qt dispatched the production paint.
                if pending is sample and sample.get('accepted_at'):
                    sample['paint_cpu_ms'] += (after - before) * 1000
                    barrier = (sample.get('records_at') if target == 'records' else
                               sample.get('summary_at') if target == 'summary' else sample['accepted_at'])
                    if barrier is not None and after >= barrier:
                        # Keep the first qualifying paint, not a later camera
                        # repaint while another widget is still catching up.
                        sample['paints'].setdefault(target, after)
                    if sample.get('summary_at') and after >= sample['summary_at']:
                        sample['post_summary_paints'].setdefault(target, after)
                return value
            return super().notify(receiver, event)

    app = App([])
    app.setQuitOnLastWindowClosed(False)
    available = app.primaryScreen().availableGeometry()
    window_size = (min(1600, available.width() - 40), min(900, available.height() - 80))
    report['requested_logical_size'] = window_size

    def instrument(method, key, owner):
        def call(self, *a, **kw):
            sample = pending
            belongs = sample is not None and current is not None and self is owner(current)
            # Timers may run before acceptance. Exclude unchanged old-summary
            # callbacks rather than inflating work with unrelated timer ticks.
            relevant = belongs and (key == 'accept_cpu_ms' or
                                    len(current['page'].session.measurements) == sample['expected_count'])
            if relevant and key == 'summary_cpu_ms':
                # Only the callback that advances statistics to this sample's
                # revision belongs to its work. Later 80 ms no-op ticks must
                # neither add CPU time nor move the paint barrier forward.
                revision = sample.get('revision')
                relevant = (revision is not None and sample.get('summary_at') is None
                            and self._last_summary_revision != revision)
            before = clock()
            result = method(self, *a, **kw)
            after = clock()
            if relevant and pending is sample:
                sample[key] += (after - before) * 1000
                if key == 'accept_cpu_ms':
                    sample['accepted_at'] = after
                    sample['revision'] = current['page'].session.revision
                    sample['record_id'] = result.id
                    sample['diameter_um'] = result.diameter_um
                elif (key == 'summary_cpu_ms' and sample.get('summary_at') is None
                      and self._last_summary_revision == sample.get('revision')):
                    sample['summary_at'] = after
                elif key == 'records_cpu_ms':
                    sample['records_at'] = after
            return result
        return call

    def distribution(values):
        values = sorted(values)
        return dict(n=len(values), p50_ms=statistics.median(values),
                    p95_ms=values[math.ceil(.95 * len(values)) - 1], max_ms=max(values))

    stack = ExitStack()
    stack.enter_context(patch.object(LiveExperimentController, '_accept',
        instrument(LiveExperimentController._accept, 'accept_cpu_ms', lambda c: c['page'].controller)))
    for name, key in (('_flush_records_notification', 'records_cpu_ms'), ('_refresh_summary', 'summary_cpu_ms')):
        stack.enter_context(patch.object(LiveExperimentWorkspace, name,
            instrument(getattr(LiveExperimentWorkspace, name), key, lambda c: c['page'])))

    try:
        for enabled in (False, True):
            harness = stack.enter_context(WorkflowHarness(output / ('profile-on' if enabled else 'profile-off'), size=window_size))
            for mode, template in templates.items():
                if template:
                    save_template_preferences({'template': str(template.resolve())}, trial_mode=mode)
            harness.enter()
            pages = {}
            for mode in ('content', 'diameter'):
                page = open_live_experiment(harness.window, embedded=True, trial_mode=mode)
                page.session.save_images = False
                page.session.configure(sample_name='完整试验 UI 性能验证', sample_number=mode)
                for direction in ('warp', 'weft'):
                    page.session.set_materials([MaterialSnapshot('wool', '绵羊毛', 1.31)], direction=direction)
                page._refresh_materials()
                page.live_excel.ensure(page.session)
                # Stop before page exposure and before Qt dispatches the first
                # manager timer. Disabled is a genuine detached production link.
                if not enabled or args.ui_only:
                    page.live_excel.stop(page.session)
                pages[mode] = page
            cases.append(dict(enabled=enabled, harness=harness, pages=pages, manager=pages['content'].live_excel))
            harness.window.hide()
        report['actual_dpr'] = [case['harness'].window.devicePixelRatioF() for case in cases]
        report['actual_logical_sizes'] = [[case['harness'].window.width(), case['harness'].window.height()] for case in cases]
        report['font'] = cases[0]['harness'].font_report()

        @safe
        def finish():
            nonlocal done, phase, pending
            pending = None
            phase = 'detach'
            if any(not case['manager'].detach(page.session) for case in cases for page in case['pages'].values()):
                QTimer.singleShot(20, finish)
                return
            if any(not case['manager'].close() for case in cases):
                QTimer.singleShot(20, finish)
                return
            for case in cases:
                assert not case['harness'].slot_errors, case['harness'].slot_errors
                assert not case['harness'].modal_messages, case['harness'].modal_messages
            for mode in ('content', 'diameter'):
                totals = {}
                for enabled in (False, True):
                    rows = [r for r in report['samples'] if r['mode'] == mode and r['enabled'] == enabled and not r['warmup']]
                    assert len(rows) == args.samples
                    totals['on' if enabled else 'off'] = {key: distribution([r[key] for r in rows]) for key in (
                        'accept_cpu_ms', 'records_cpu_ms', 'summary_cpu_ms', 'callbacks_cpu_ms',
                        'paint_cpu_ms', 'finish_to_accept_ms', 'finish_to_ui_ms', 'finish_call_ms')}
                totals['ratios'] = {key: totals['on'][key]['p95_ms'] / totals['off'][key]['p95_ms']
                                    for key in ('accept_cpu_ms', 'callbacks_cpu_ms', 'finish_to_ui_ms')}
                totals['within_10_percent'] = {key: ratio <= 1.1 for key, ratio in totals['ratios'].items()}
                report['metrics'][mode] = totals
                left, right = (case['pages'][mode].session for case in cases)
                assert [r.diameter_um for r in left.measurements.values()] == [r.diameter_um for r in right.measurements.values()]
            report['passed'] = True  # Correctness/completion only; performance gates are reported separately.
            report['elapsed_ms'] = (clock() - started) * 1000
            persist()
            done = True
            app.exit(0)

        @safe
        def next_case():
            nonlocal current, phase, pair_number, pair_order, pair_position, mode_index, pending
            if pair_position >= len(pair_order):
                pair_number += 1
                pair_position = 0
                if pair_number > args.samples + args.warmup:
                    mode_index += 1
                    pair_number = 1
                if mode_index >= 2:
                    finish()
                    return
                pair_order = [0, 1] if pair_number % 2 else [1, 0]
            case = cases[pair_order[pair_position]]
            mode = ('content', 'diameter')[mode_index]
            current = {**case, 'mode': mode, 'page': case['pages'][mode]}
            pending = None
            phase = 'expose'
            # Exposure and camera readiness are entirely outside sample timing.
            for item in cases:
                item['harness'].window.setVisible(item is case)
            harness, page = case['harness'], current['page']
            harness.window.raise_()
            harness.window.activateWindow()
            harness.hub.select_page(mode_index + 1)
            page.controller.set_calibration(CalibrationSnapshot(1, 1, 1, 'UI benchmark'))
            page.display_binding.set_mode('fit')
            harness.hub.records.open(page)
            harness.hub.records.set_position('bottom')
            current['expose_at'] = clock()

        @safe
        def begin_sample():
            nonlocal pending, phase
            page = current['page']
            count = len(page.session.measurements)
            assert count == pair_number - 1, (count, pair_number)
            table = current['harness'].hub.records.panel.table
            assert table.isVisible() and page.canvas.isVisible() and page.latest_value.isVisible()
            summary_widget = (page.material_rows['wool']['measured_count'] if current['mode'] == 'content'
                              else page.diameter_summary)
            assert summary_widget.isVisible(), 'The statistics display must be visible in the measurement'
            # Coordinates are inside the shared fake-camera frame. No image
            # evidence writes or boundary confirmation changes the UI workload.
            start = (300., 220.)
            end = (300., 240. + (pair_number % 10) / 10)
            page.controller.start(start)
            t0 = clock()
            pending = dict(mode=current['mode'], enabled=current['enabled'], warmup=pair_number <= args.warmup,
                pair=pair_number, t0=t0, expected_count=count + 1, accepted_at=None, summary_at=None,
                records_at=None, accept_cpu_ms=0., records_cpu_ms=0., summary_cpu_ms=0.,
                paint_cpu_ms=0., paints={}, post_summary_paints={},
                paint_targets={id(page.canvas): 'canvas', id(table.viewport()): 'records',
                               id(page.latest_value): 'value', id(summary_widget): 'summary'})
            page.controller.finish(end, 'wool', method='free')
            pending['finish_call_ms'] = (clock() - t0) * 1000
            phase = 'sample'

        @safe
        def poll():
            nonlocal phase, pending, pair_position
            if done or current is None:
                return
            page, manager = current['page'], current['manager']
            state = manager.status(page.session.id)
            if state.get('excel_pid'):
                report['workbooks'][page.session.id] = {key: state.get(key) for key in (
                    'name', 'path', 'excel_pid', 'excel_created', 'save_state', 'generation')}
            if current['enabled'] and not args.ui_only and state['state'] in ('error', 'unsupported', 'stopped'):
                raise RuntimeError(state)
            if phase == 'expose':
                ready = (page.controller.state == 'ready' and page.controller.presented is not None
                         and page._last_summary_revision == page.session.revision)
                if current['enabled'] and not args.ui_only:
                    ready = ready and state.get('connected') and not state.get('pending')
                else:
                    ready = ready and state['state'] == 'detached'
                if ready:
                    begin_sample()
                elif clock() - current['expose_at'] > 45:
                    raise RuntimeError(dict(reason='Exposure/initialization timeout', controller=page.controller.state, excel=state))
                return
            if phase == 'sample':
                sample = pending
                if clock() - sample['t0'] > 15:
                    raise RuntimeError(dict(reason='Measurement UI timeout', sample=sample,
                        pending_classification=len(page.controller.pending_classification), jobs=len(page.controller._jobs)))
                records = current['harness'].hub.records.panel
                ready = (sample['accepted_at'] is not None and sample['summary_at'] is not None
                         and sample['records_at'] is not None and records.proxy.rowCount() == sample['expected_count']
                         and all(key in sample['paints'] for key in ('canvas', 'records', 'value'))
                         and sample['paints'].get('records', 0) >= sample['records_at']
                         and 'summary' in sample['post_summary_paints'])
                if not ready:
                    return
                # All boundaries are actual callback/paint timestamps; the
                # 1 ms observer polling interval is not added to elapsed times.
                end = max(sample['summary_at'], sample['records_at'], *sample['paints'].values())
                sample['finish_to_ui_ms'] = (end - sample['t0']) * 1000
                sample['finish_to_accept_ms'] = (sample['accepted_at'] - sample['t0']) * 1000
                sample['callbacks_cpu_ms'] = sum(sample[key] for key in ('accept_cpu_ms', 'records_cpu_ms', 'summary_cpu_ms'))
                sample.pop('paint_targets')
                report['samples'].append(sample)
                pending = None
                phase = 'sync_barrier'
                current['last_revision'] = sample['revision']
                return
            if phase == 'sync_barrier':
                # Native writes must finish before the next pair, but this is
                # NOT inside any UI response metric and never calls flush().
                if current['enabled'] and not args.ui_only and (state.get('pending') or state.get('input_revision') != current['last_revision']):
                    return
                report['operations'].append(dict(mode=current['mode'], enabled=current['enabled'], pair=pair_number,
                    input_revision=state.get('input_revision'), excel_pid=state.get('excel_pid'),
                    excel_created=state.get('excel_created'), state=state['state'],
                    response={key: state.get(key) for key in ('name', 'path', 'excel_pid', 'excel_created',
                                                             'save_state', 'input_revision')}))
                if pair_number % 20 == 0:
                    persist()
                pair_position += 1
                QTimer.singleShot(0, next_case)
                phase = 'between'

        @safe
        def camera_tick():
            if current is not None and phase not in ('detach', 'between'):
                backend = current['harness'].backend
                if backend.callbacks:
                    # Unlike harness.tick(), this never pumps a nested loop.
                    backend.sequence += 1
                    backend.callbacks[-1][0](frame)

        frame = synthetic_frame()
        camera = QTimer()
        camera.setInterval(33)
        camera.timeout.connect(camera_tick)
        camera.start()
        observer = QTimer()
        observer.setInterval(1)
        observer.timeout.connect(poll)
        observer.start()
        watchdog = QTimer()
        watchdog.setSingleShot(True)
        watchdog.timeout.connect(lambda: fail(RuntimeError('UI benchmark watchdog expired')))
        watchdog.start(args.timeout * 1000)
        QTimer.singleShot(0, next_case)
        app.exec()
        camera.stop()
        observer.stop()
        watchdog.stop()
    except Exception as exc:
        fail(exc)
    finally:
        # Parent owns disposable Excel cleanup by returned PID + creation time.
        # Force-release on failure avoids lifecycle dialogs masking the evidence.
        if not report['passed']:
            for case in cases:
                case['manager'].close()
        stack.close()
    print(json.dumps(dict(passed=report['passed'], native_excel=report['native_excel'],
                         samples=len(report['samples']), metrics=report['metrics'], errors=report['errors']), ensure_ascii=False))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
