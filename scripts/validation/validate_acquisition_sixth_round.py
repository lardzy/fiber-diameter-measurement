"""Round-six real-navigation regression and bounded scheduling probes.

The source is synthetic 768x576 pixels. Navigation, automatic analysis process,
strict alignment, canvas clicks, ledger and evidence writes are production paths.
Delay hooks simulate GUI consumption versus worker completion separately; they
never inject candidates, motion decisions, alignment decisions or accepted rows.
Qt-grab video is diagnostic and is not a native desktop/camera performance run.
"""
from __future__ import annotations

import argparse
import cProfile
from dataclasses import asdict, replace
from datetime import datetime, timezone
import functools
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import pstats
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario', choices=('cycle', 'clicks', 'latency'), default='cycle')
    p.add_argument('--duration', type=float, default=60)
    p.add_argument('--clicks', type=int, default=100)
    p.add_argument('--delay-kind', choices=('gui', 'worker'), default='gui')
    p.add_argument('--delay-ms', type=int, choices=(0, 40, 80, 200), default=0)
    p.add_argument('--theme', choices=('light', 'dark', 'navy', 'warm'), default='light')
    p.add_argument('--size', default='1600x900')
    p.add_argument('--scale', type=float, default=1)
    p.add_argument('--base-device-scale', type=float, default=1)
    p.add_argument('--platform', default='windows' if sys.platform == 'win32' else 'offscreen')
    p.add_argument('--video', action='store_true')
    p.add_argument('--profile-clicks', action='store_true',
                   help='Profile QTest mouse clicks; diagnostic overhead, not performance acceptance.')
    p.add_argument('--trace-level', choices=('detailed', 'light'), default='detailed',
                   help='Light keeps transitions, clicks, rejects and slow-GC/pause evidence; no per-frame/method trace.')
    p.add_argument('--output', required=True, type=Path)
    return p.parse_args()


def main(args):
    os.environ['QT_QPA_PLATFORM'] = args.platform
    os.environ['QT_SCALE_FACTOR'] = str(args.scale / args.base_device_scale)
    sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts/validation')]
    import cv2
    import numpy as np
    from PySide6.QtCore import QEventLoop, QPoint, Qt, QTimer
    from PySide6.QtGui import QImage
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QDialogButtonBox, QPushButton, QTableWidget
    from validate_workflow_redesign import WorkflowHarness, dimensions, synthetic_frame
    from validate_acquisition_fifth_round import fingerprints
    from fdm.live_experiment.candidate_alignment import CandidateAlignmentMonitor, compare_candidate_frames
    from fdm.live_experiment.controller import LiveExperimentController
    from fdm.live_experiment import ExperimentSession
    from fdm.live_experiment.motion import compare_geometry, reduced_gray, sobel_fraction
    from fdm.ui.live_measurement_canvas import LiveMeasurementCanvas
    from fdm.ui.live_experiment_workspace import LiveExperimentWorkspace

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'result.json').exists():
        raise FileExistsError('Preserve prior result; select a new output directory')
    profile = Path(tempfile.mkdtemp(prefix='profile-', dir=output))
    physical = dimensions(args.size)
    logical = tuple(round(x / args.scale) for x in physical)
    started = time.perf_counter()
    trace = dict(analyses=[], visibility=[], status=[], alignments=[], delays=[], actions=[], samples=[],
                 click_attempts=[], pointer_events=[], gesture_rejections=[], candidate_signals=[],
                 presentation_attempts=[], synchronous_costs=[], gc=[], slow_gc=[], gc_count=0, captures=[])
    report = dict(schema_version=1, started_at=datetime.now(timezone.utc).isoformat(),
        scenario=args.scenario, duration=args.duration, requested_clicks=args.clicks,
        delay_kind=args.delay_kind, delay_ms=args.delay_ms, theme=args.theme,
        requested_client_raster=physical, scale=args.scale, runtime=platform.platform(),
        platform=args.platform, real_camera=False, source_fingerprints=fingerprints(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), passed=False,
        scope='Real navigation, analysis process, alignment, UI clicks, ledger and JPEG. Synthetic camera.',
        trace=trace)
    injection = dict(enabled=False, stage='setup')
    original_work = CandidateAlignmentMonitor._work
    original_paint = LiveMeasurementCanvas.paintEvent
    original_pointer_begin = LiveMeasurementCanvas._begin_pointer_gesture
    original_pointer_finish = LiveMeasurementCanvas._finish_pointer_gesture
    recording = dict(process=None, timer=None, frames=0, errors=[])
    profiler = cProfile.Profile() if args.profile_clicks else None
    timed_methods = []
    gc_starts = {}
    main_thread = threading.get_ident()
    report['instrumentation'] = dict(
        scope='Bounded perf_counter method, GC and capture timings; no product or GC-policy changes.',
        cprofile_enabled=bool(profiler),
        trace_level=args.trace_level,
        performance_acceptance=False if profiler else None,
        gc_threshold=gc.get_threshold(),
        method_capacity=8192, gc_capacity=512, slow_gc_capacity=128)
    h = None
    ticker = None

    def now():
        return round(time.perf_counter() - started, 6)

    def record_gc(phase, info):
        key = (threading.get_ident(), info['generation'])
        if phase == 'start':
            gc_starts[key] = time.perf_counter()
        elif phase == 'stop':
            stopped = time.perf_counter()
            began = gc_starts.pop(key, None)
            if began is not None:
                trace['gc_count'] += 1
                if args.trace_level == 'light' and key[1] != 2 and (stopped-began)*1000 < 2:
                    return
                row = dict(at=round(began-started, 6),
                    end_at=round(stopped-started, 6), elapsed_ms=(stopped-began)*1000,
                    generation=key[1], main_thread=key[0] == main_thread,
                    collected=info.get('collected'), uncollectable=info.get('uncollectable'),
                    stage=injection['stage'])
                trace['gc'].append(row)
                del trace['gc'][:-512]
                if key[1] == 2 or row['elapsed_ms'] >= 2:
                    trace['slow_gc'].append(row)
                    del trace['slow_gc'][:-128]

    def time_method(cls, name):
        original = getattr(cls, name)
        own_method = name in cls.__dict__
        @functools.wraps(original)
        def timed(instance, *values, **kwargs):
            if injection['stage'] != 'clicks':
                return original(instance, *values, **kwargs)
            began = time.perf_counter()
            try:
                return original(instance, *values, **kwargs)
            finally:
                stopped = time.perf_counter()
                trace['synchronous_costs'].append(dict(at=round(began-started, 6),
                    end_at=round(stopped-started, 6), elapsed_ms=(stopped-began)*1000,
                    method=f'{cls.__name__}.{name}',
                    main_thread=threading.get_ident() == main_thread,
                    last_click_at=trace['click_attempts'][-1]['at'] if trace['click_attempts'] else None))
                del trace['synchronous_costs'][:-8192]
        setattr(cls, name, timed)
        timed_methods.append((cls, name, original, own_method))

    def timed_capture(label):
        began = time.perf_counter()
        image = h.window.grab()
        stopped = time.perf_counter()
        trace['captures'].append(dict(at=round(began-started, 6), end_at=round(stopped-started, 6),
                                     elapsed_ms=(stopped-began)*1000, operation=label))
        return image

    def pause(ms):
        loop = QEventLoop()
        timer = QTimer()
        timer.setTimerType(Qt.TimerType.PreciseTimer)
        timer.setSingleShot(True)
        timer.timeout.connect(loop.quit)
        timer.start(max(1, round(ms)))
        loop.exec()

    def delayed_work(monitor, *values, **kwargs):
        value = original_work(monitor, *values, **kwargs)
        if args.delay_kind == 'worker' and injection['enabled'] and args.delay_ms:
            before = time.perf_counter()
            time.sleep(args.delay_ms / 1000)
            elapsed = (time.perf_counter() - before) * 1000
            trace['delays'].append(dict(at=now(), kind='worker', elapsed_ms=elapsed))
            value = replace(value, elapsed_ms=value.elapsed_ms + elapsed)
        return value

    def painted(canvas, event):
        original_paint(canvas, event)
        if h is None or canvas is not getattr(page, 'canvas', None):
            return
        ctx = canvas.controller._automatic_context
        value = bool(canvas._candidates_visible())
        batch = ctx.get('batch_id') if ctx else None
        available = sum(x.get('state') in ('available', 'confirming') for x in canvas.controller.candidates)
        previous = trace['visibility'][-1] if trace['visibility'] else None
        if (previous is None or previous['visible'] != value or previous['batch'] != batch
                or previous['available'] != available or previous['stage'] != injection['stage']):
            trace['visibility'].append(dict(at=now(), visible=value, stage=injection['stage'],
                batch=batch, available=available, frame=canvas.frame.sequence if canvas.frame else None))

    def pointer_snapshot(canvas):
        token = canvas.controller.presented
        gesture = canvas._pointer_gesture
        popup = QApplication.activePopupWidget()
        return dict(at=now(), frame=canvas.frame.sequence if canvas.frame else None,
            presented=token.frame.sequence if token else None,
            same_frame=bool(token is not None and token.frame is canvas.frame),
            display_revision=canvas.display_revision,
            token_revision=token.display_revision if token else None,
            pointer_id=gesture['pointer_id'] if gesture else None,
            pointer_kind=gesture['kind'] if gesture else None,
            pointer_current=canvas._pointer_current(gesture) if gesture else None,
            candidates=list(gesture['candidates']) if gesture else [],
            candidate_visible=canvas._candidates_visible(),
            popup=type(popup).__name__ if popup else None)

    def observed_pointer_begin(canvas, event):
        if page is None or canvas is not page.canvas:
            return original_pointer_begin(canvas, event)
        before = pointer_snapshot(canvas)
        result = original_pointer_begin(canvas, event)
        trace['pointer_events'].append(dict(kind='press', x=event.position().x(),
            y=event.position().y(), before=before, after=pointer_snapshot(canvas)))
        return result

    def observed_pointer_finish(canvas, event):
        if page is None or canvas is not page.canvas:
            return original_pointer_finish(canvas, event)
        before = pointer_snapshot(canvas)
        result = original_pointer_finish(canvas, event)
        trace['pointer_events'].append(dict(kind='release', x=event.position().x(),
            y=event.position().y(), before=before, after=pointer_snapshot(canvas)))
        return result

    CandidateAlignmentMonitor._work = delayed_work
    LiveMeasurementCanvas.paintEvent = painted
    if args.trace_level == 'detailed':
        LiveMeasurementCanvas._begin_pointer_gesture = observed_pointer_begin
        LiveMeasurementCanvas._finish_pointer_gesture = observed_pointer_finish
    if args.scenario == 'clicks' and args.trace_level == 'detailed':
        for cls, names in (
            (LiveExperimentController, ('submit_candidate', 'accept_candidates', '_commit_candidates',
                                        '_queue_confirmation', '_reserve_evidence')),
            (ExperimentSession, ('add_measurements_batch', '_prepare_measurement',
                                 '_preflight_measurements', '_commit', '_apply',
                                 '_touch', '_record_delta')),
            (LiveExperimentWorkspace, ('_records_accepted', '_record_accepted', '_changed',
                                       '_candidates_changed', '_refresh_assistance_state',
                                       '_refresh_summary', '_queue_records_notification',
                                       '_flush_records_notification', '_flush_evidence_refresh',
                                       '_check_reminders'))):
            for name in names:
                time_method(cls, name)
    gc.callbacks.append(record_gc)
    page = None
    try:
        with WorkflowHarness(profile, theme=args.theme, size=logical) as h:
            h.enter()
            page = h.select(1, frame=False)
            c = page.controller
            page.canvas.gestureRejected.connect(lambda reason: trace['gesture_rejections'].append(dict(at=now(), reason=reason)))
            page.canvas.candidateClicked.connect(lambda key: trace['candidate_signals'].append(dict(at=now(), kind='clicked', id=key)))
            page.canvas.candidateAmbiguous.connect(lambda keys: trace['candidate_signals'].append(dict(at=now(), kind='ambiguous', ids=keys)))
            back_off = c._back_off_candidate_preview
            def observed_back_off(at, **kwargs):
                sampled_at = time.perf_counter()
                pending = c._candidate_preview_pending
                checked = c._alignment.completed_frame
                submitted_at = c._alignment.completed_submitted_at
                ready_at = c._alignment.completed_ready_at
                future = c._alignment.future
                trace['presentation_attempts'].append(dict(at=now(),
                    caller=sys._getframe(1).f_code.co_name,
                    reason=kwargs.get('reason', 'verification_late'),
                    already_paused=bool(c._candidate_preview_backoff_until),
                    pending_sequence=pending.sequence if pending else None,
                    pending_age_ms=(sampled_at-pending.received_at)*1000 if pending else None,
                    deadline_overdue_ms=(sampled_at-c._candidate_preview_deadline)*1000
                        if c._candidate_preview_deadline is not None else None,
                    checked_sequence=checked.sequence if checked else None,
                    checked_age_ms=(sampled_at-checked.received_at)*1000 if checked else None,
                    verification_ready_ms=(ready_at-submitted_at)*1000
                        if ready_at is not None and submitted_at is not None else None,
                    gui_wait_ms=(sampled_at-ready_at)*1000 if ready_at is not None else None,
                    future_sequence=future[3] if future else None,
                    future_done=future[0].done() if future else None,
                    future_submitted_ms=(sampled_at-future[7])*1000 if future else None,
                    latest_sequence=c.latest_frame.sequence if c.latest_frame else None,
                    last_click_at=trace['click_attempts'][-1]['at'] if trace['click_attempts'] else None,
                    recent_gc=list(trace['gc'][-10:]),
                    wait_budget_ms=c._candidate_preview_wait_budget()*1000))
                del trace['presentation_attempts'][:-512]
                return back_off(at, **kwargs)
            c._back_off_candidate_preview = observed_back_off
            source = dict(image=synthetic_frame(sequence=0))
            ticker = QTimer(h.window)
            ticker.setTimerType(Qt.TimerType.PreciseTimer)
            ticker.setInterval(33)
            ticker.timeout.connect(lambda: h.backend.tick(image=source['image']))
            ticker.start()
            c.statusChanged.connect(lambda msg: trace['status'].append(dict(at=now(), message=msg)))
            analyze = c.analyze
            def observed_analyze(*values, **kwargs):
                result = analyze(*values, **kwargs)
                ctx = c._automatic_context
                trace['analyses'].append(dict(at=now(), origin=kwargs.get('_origin', 'manual'),
                    batch=ctx.get('batch_id') if ctx else None, stage=injection['stage']))
                return result
            c.analyze = observed_analyze
            # Observe genuine completed comparisons, preserving all decisions.
            poll_alignment = c._alignment.poll
            def observed_alignment(*values, **kwargs):
                result = poll_alignment(*values, **kwargs)
                if result:
                    consumed_at = time.perf_counter()
                    submitted_at = c._alignment.completed_submitted_at
                    ready_at = c._alignment.completed_ready_at
                    checked = c._alignment.completed_frame
                    # Avoid thousands of retained timing dictionaries in the
                    # functional acceptance path. Detailed timing is a
                    # separate diagnostic workload and can perturb GC.
                    timely = (result[3].valid and ready_at is not None and submitted_at is not None
                        and ready_at-submitted_at <= c._candidate_preview_wait_budget()
                        and consumed_at-ready_at <= c._candidate_preview_wait_budget())
                    if args.trace_level == 'light' and timely:
                        return result
                    trace['alignments'].append(dict(at=now(), valid=result[3].valid,
                        reason=result[3].reason, sequence=result[1], stage=injection['stage'],
                        algorithm_ms=result[3].elapsed_ms,
                        verification_ready_ms=(ready_at-submitted_at)*1000
                            if ready_at is not None and submitted_at is not None else None,
                        gui_wait_ms=(consumed_at-ready_at)*1000 if ready_at is not None else None,
                        frame_age_ms=(consumed_at-checked.received_at)*1000 if checked else None,
                        wait_budget_ms=c._candidate_preview_wait_budget()*1000))
                    del trace['alignments'][:-4096]
                return result
            c._alignment.poll = observed_alignment

            def wait(predicate, label, timeout=12):
                deadline = time.perf_counter() + timeout
                while not predicate() and time.perf_counter() < deadline:
                    pause(10)
                if not predicate():
                    report['failure_snapshot'] = pointer_snapshot(page.canvas)
                    report['input_trace'] = c.performance_frame_context().get('input_trace')
                    report['presentation_events'] = list(c._candidate_presentation_events)
                    report['automatic_cycle'] = c.performance_frame_context().get('automatic_cycle')
                    report['accepted_total'] = len(page.session.measurements)
                    raise AssertionError(dict(label=label, status=page.statusBar().currentMessage(),
                        field=c.field_epoch, state=c.state, candidates=len(c.candidates),
                        trace=trace['visibility'][-10:], errors=h.slot_errors, modals=h.modal_messages,
                        diagnostics=c.performance_diagnostics()))

            def rename():
                dialog = QApplication.activeModalWidget()
                dialog.setTextValue('Operator A')
                dialog.accept()
            QTimer.singleShot(0, rename)
            page.rename_operator_button.trigger()
            def categories():
                dialog = QApplication.activeModalWidget()
                table = dialog.findChild(QTableWidget)
                for row in range(3):
                    table.cellWidget(row, 0).setCurrentIndex(row + 1)
                    table.cellWidget(row, 3).setValue(0)
                    table.cellWidget(row, 4).setValue(0)
                QTest.mouseClick(dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Ok), Qt.MouseButton.LeftButton)
            QTimer.singleShot(0, categories)
            QTest.mouseClick(page.choose_categories_button, Qt.MouseButton.LeftButton)
            page.ruler_choice.setFocus()
            for key in (Qt.Key.Key_Home, Qt.Key.Key_Down, Qt.Key.Key_Tab):
                QTest.keyClick(page.ruler_choice, key)
            wait(lambda: c.state == 'ready' and page.analyze_button.isEnabled(), 'ready')
            assert page.input_mode.currentData() == 'mixed'
            assert not page.auto_analysis.isChecked()
            QTest.mouseClick(page.auto_analysis, Qt.MouseButton.LeftButton, pos=QPoint(8, page.auto_analysis.height() // 2))
            wait(lambda: bool(c.candidates) and page.canvas._candidates_visible(), 'first automatic candidates')
            initial_batch = c._automatic_context['batch_id']
            initial_ids = [item['candidate_id'] for item in c.candidates]
            pause(600)
            report['initial_candidates'] = len(initial_ids)
            report['actual_client_raster'] = [timed_capture('initial-width-grab').width(),
                                              timed_capture('initial-height-grab').height()]
            assert report['actual_client_raster'] == list(physical)
            initial_image = timed_capture('initial-screenshot-grab')
            save_started = time.perf_counter()
            initial_image.save(str(output / 'initial.png'))
            trace['captures'].append(dict(at=round(save_started-started, 6), end_at=now(),
                elapsed_ms=(time.perf_counter()-save_started)*1000, operation='initial-png-save'))
            del initial_image
            if args.video:
                ffmpeg = shutil.which('ffmpeg')
                if ffmpeg is None:
                    raise RuntimeError('--video requires ffmpeg')
                recording['process'] = subprocess.Popen([ffmpeg, '-y', '-loglevel', 'error', '-f', 'rawvideo',
                    '-pixel_format', 'rgba', '-video_size', f'{physical[0]}x{physical[1]}', '-framerate', '8', '-i', '-',
                    '-an', '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '24', '-pix_fmt', 'yuv420p',
                    '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', str(output / 'workflow.mp4')],
                    stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                recording['timer'] = QTimer(h.window)
                recording['timer'].setInterval(125)
                def record():
                    try:
                        image = h.window.grab().toImage().convertToFormat(QImage.Format.Format_RGBA8888)
                        recording['process'].stdin.write(image.constBits())
                        recording['frames'] += 1
                    except Exception as error:
                        recording['errors'].append(str(error))
                        recording['timer'].stop()
                recording['timer'].timeout.connect(record)
                recording['timer'].start()

            if args.scenario == 'clicks':
                # Initial QWidget grabs/PNG encoding are deliberately
                # synchronous diagnostic work, not user measurement load.
                # Preserve their pauses in setup, then let strict recovery
                # finish before evaluating uninterrupted repeated clicks.
                pause(600)
                wait(lambda: page.canvas._candidates_visible()
                     and c.presented is not None and c.presented.frame is page.canvas.frame,
                     'initial capture recovery')
                report['capture_recovery_finished_at'] = now()
            new_field = next(button for button in page.findChildren(QPushButton) if button.text() == '新视场')
            if args.scenario == 'cycle':
                injection['stage'] = 'static'
                stable_start = now()
                deadline = time.perf_counter() + args.duration
                while time.perf_counter() < deadline:
                    pause((deadline - time.perf_counter()) * 1000 + 1)
                assert len(trace['analyses']) == 1, trace['analyses']
                assert c._automatic_context['batch_id'] == initial_batch
                report['static'] = dict(seconds=now() - stable_start, analyses=len(trace['analyses']),
                    source_frames=h.backend.sequence, batch=initial_batch)
                # Find a real blurred input that fails fixed-reference alignment
                # but remains stable under the unchanged global field detector.
                rgba = source['image'].convertToFormat(QImage.Format.Format_RGBA8888)
                pixels = np.frombuffer(rgba.constBits(), np.uint8).reshape(rgba.height(), rgba.width(), 4).copy()
                blur = None
                for sigma in (.8, 1, 1.2, 1.4, 1.6, 2):
                    candidate = cv2.GaussianBlur(pixels, (0, 0), sigma)
                    check = compare_candidate_frames(pixels, candidate)
                    ga, gb = reduced_gray(pixels), reduced_gray(candidate)
                    geometry = compare_geometry(ga, gb)
                    fraction = sobel_fraction(ga, gb)
                    if not check.valid and geometry[0] == 'stable':
                        blur = candidate
                        report['blur_fixture'] = dict(sigma=sigma, alignment=asdict(check),
                            global_geometry=geometry, changed_fraction=fraction)
                        break
                assert blur is not None, 'No independent alignment-failure image found'
                epoch = c.field_epoch
                injection['stage'] = 'blur'
                ramp = [cv2.GaussianBlur(pixels, (0, 0), float(value)) for value in np.arange(.1, sigma + .05, .1)]
                previous = pixels
                report['blur_fixture']['ramp_fractions'] = []
                for image in ramp:
                    fraction = sobel_fraction(reduced_gray(previous), reduced_gray(image))
                    assert fraction <= .04138815884775365
                    report['blur_fixture']['ramp_fractions'].append(fraction)
                    source['image'] = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format.Format_RGBA8888).copy()
                    pause(80)
                    assert c.field_epoch == epoch and c.state == 'ready'
                    previous = image
                pause(1500)
                assert c.state == 'ready' and c.field_epoch == epoch
                assert len(trace['analyses']) == 1
                assert c._automatic_context['batch_id'] == initial_batch
                assert [x['candidate_id'] for x in c.candidates] == initial_ids
                assert not page.canvas._candidates_visible()
                h.window.grab().save(str(output / 'alignment-paused.png'))
                injection['stage'] = 'restored'
                for image in reversed(ramp[:-1]):
                    source['image'] = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format.Format_RGBA8888).copy()
                    pause(80)
                    assert c.field_epoch == epoch and c.state == 'ready'
                source['image'] = synthetic_frame(sequence=0)
                wait(page.canvas._candidates_visible, 'same batch strict alignment recovered')
                assert c._automatic_context['batch_id'] == initial_batch and len(trace['analyses']) == 1
                QTest.mouseClick(new_field, Qt.MouseButton.LeftButton)
                wait(lambda: c._automatic_context is not None and c._automatic_context['batch_id'] != initial_batch
                     and page.canvas._candidates_visible(), 'explicit new field one analysis')
                pause(800)
                assert len(trace['analyses']) == 2
                report['cycle'] = dict(paused_same_field=True, recovered_same_batch=True, explicit_new_field_analyses=2)
            elif args.scenario == 'clicks':
                injection['stage'] = 'clicks'
                accepted = 0
                batches = []
                while accepted < args.clicks:
                    wait(lambda: bool(c.candidates) and page.canvas._candidates_visible(), 'clickable batch')
                    batch = c._automatic_context['batch_id']
                    ids = [x['candidate_id'] for x in c.candidates]
                    batches.append(batch)
                    while accepted < args.clicks:
                        available = [item for item in c.candidates if item.get('state') == 'available']
                        if not available:
                            break
                        item = available[0]
                        scale, offset = page.canvas._transform()
                        center = tuple((a+b)/2 for a,b in zip(item['start'], item['end']))
                        pos = QPoint(round(center[0]*scale+offset[0]), round(center[1]*scale+offset[1]))
                        page.image_scroll.ensureVisible(pos.x(), pos.y(), 20, 20)
                        # Input is valid only after the candidate-bearing frame
                        # has actually painted. A queued set_frame can precede
                        # paint; do not synthesize clicks into that safety gap.
                        wait(lambda: page.canvas._candidates_visible()
                             and c.presented is not None
                             and c.presented.frame is page.canvas.frame, 'painted click gate')
                        before = len(page.session.measurements)
                        action = dict(at=now(), candidate=item['candidate_id'], batch=batch, before=before)
                        attempt = dict(**action, position=[pos.x(), pos.y()])
                        if args.trace_level == 'detailed':
                            attempt['presentation'] = pointer_snapshot(page.canvas)
                        trace['click_attempts'].append(attempt)
                        if profiler:
                            profiler.enable()
                        try:
                            QTest.mouseClick(page.canvas, Qt.MouseButton.LeftButton, pos=pos)
                        finally:
                            if profiler:
                                profiler.disable()
                        wait(lambda: len(page.session.measurements) == before + 1, 'candidate accepted')
                        action['accepted_at'] = now()
                        action['elapsed_ms'] = (action['accepted_at']-action['at'])*1000
                        trace['actions'].append(action)
                        accepted += 1
                        assert [x['candidate_id'] for x in c.candidates] == ids
                        assert c._automatic_context['batch_id'] == batch
                        assert page.workflow_tabs.currentIndex() == 0
                        pause(120)
                    if accepted < args.clicks:
                        QTest.mouseClick(new_field, Qt.MouseButton.LeftButton)
                        wait(lambda: c._automatic_context is not None and c._automatic_context['batch_id'] != batch
                             and page.canvas._candidates_visible(), 'next explicit field')
                pause(1000)
                wait(lambda: all(row.evidence.get('state') == 'saved' for row in page.session.measurements.values()), 'all evidence saved')
                # Batch/availability transitions are expected. A same-batch
                # visible->hidden while other candidates remain is not.
                hides = [b for a,b in zip(trace['visibility'], trace['visibility'][1:])
                    if a['visible'] and not b['visible'] and a['batch'] == b['batch']
                    and b['available'] and b['stage'] == 'clicks']
                report['clicks'] = dict(accepted=accepted, batches=len(batches),
                    remaining_candidate_hides=hides, records=len(page.session.measurements),
                    saved=sum(row.evidence.get('state') == 'saved' for row in page.session.measurements.values()))
                # Preserve scheduling evidence even when the visibility
                # assertion fails after all records were correctly accepted.
                report['performance'] = c.performance_diagnostics()
                report['presentation_events'] = list(c._candidate_presentation_events)
                report['input_trace'] = c.performance_frame_context().get('input_trace')
                report['automatic_cycle'] = c.performance_frame_context().get('automatic_cycle')
                assert not hides, hides
            else:
                injection['stage'] = 'latency'
                injection['enabled'] = True
                for _ in range(6):
                    if args.delay_kind == 'gui':
                        # Event-loop sleep delays GUI consumption only; alignment
                        # completion timestamps are stamped on its worker thread.
                        t = time.perf_counter()
                        time.sleep(args.delay_ms/1000)
                        trace['delays'].append(dict(at=now(), kind='gui', elapsed_ms=(time.perf_counter()-t)*1000))
                    pause(150)
                injection['enabled'] = False
                injection['stage'] = 'recovery'
                wait(page.canvas._candidates_visible, 'latency recovery', timeout=10)
                pause(1000)
                assert c._automatic_context['batch_id'] == initial_batch
                assert len(trace['analyses']) == 1
                report['latency'] = dict(analysis_count=1, batch_preserved=True,
                    presentation_events=list(c._candidate_presentation_events))
            h.window.grab().save(str(output / 'final.png'))
            report['performance'] = c.performance_diagnostics()
            report['automatic_cycle'] = c.performance_frame_context().get('automatic_cycle')
            report['presentation_events'] = list(c._candidate_presentation_events)
            report['input_trace'] = c.performance_frame_context().get('input_trace')
            report['accepted_total'] = len(page.session.measurements)
            assert not h.slot_errors, h.slot_errors
            assert not h.modal_messages, h.modal_messages
            report['passed'] = True
            ticker.stop()
            if recording['timer'] is not None:
                recording['timer'].stop()
    except Exception:
        report['error'] = traceback.format_exc()
        if h is not None:
            report['slot_errors'] = h.slot_errors
            report['modal_messages'] = h.modal_messages
    finally:
        CandidateAlignmentMonitor._work = original_work
        LiveMeasurementCanvas.paintEvent = original_paint
        LiveMeasurementCanvas._begin_pointer_gesture = original_pointer_begin
        LiveMeasurementCanvas._finish_pointer_gesture = original_pointer_finish
        if record_gc in gc.callbacks:
            gc.callbacks.remove(record_gc)
        for cls, name, original, own_method in reversed(timed_methods):
            if own_method:
                setattr(cls, name, original)
            else:
                delattr(cls, name)
        if profiler:
            stream = io.StringIO()
            pstats.Stats(profiler, stream=stream).sort_stats('cumulative').print_stats(50)
            (output / 'click-profile.txt').write_text(stream.getvalue(), encoding='utf-8')
            profiler.dump_stats(str(output / 'click-profile.pstats'))
        if recording['process'] is not None:
            process = recording['process']
            process.stdin.close()
            code = process.wait(timeout=30)
            report['video'] = dict(frames=recording['frames'], fps=8, kind='Qt widget grabs; diagnostic overhead',
                exit_code=code, errors=recording['errors'], stderr=process.stderr.read().decode(errors='replace'))
            if code or recording['errors']:
                report['passed'] = False
        report['source_changed_during_run'] = fingerprints() != report['source_fingerprints']
        report['elapsed_seconds'] = now()
        (output / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'], output=str(output), error=report.get('error')), ensure_ascii=False))
    return int(not report['passed'])


if __name__ == '__main__':
    raise SystemExit(main(arguments()))
