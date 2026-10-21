"""Round-seven real-navigation checks, separated into quick and full phases.

Uses a synthetic 768x576 camera with the actual MainWindow, analysis process,
field detector, canvas pointer routes, ledger and evidence writer. No candidate,
motion decision, accepted record or rendered-frame token is injected. Optional
latency delays preserve the actual global detector result. This is not physical
Microview acceptance. Each output directory is immutable once result.json exists.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def fingerprints():
    paths = sorted((ROOT / 'src/fdm/live_experiment').glob('*.py'))
    paths += [ROOT / 'src/fdm/ui' / name for name in (
        'live_measurement_canvas.py', 'live_experiment_workspace.py',
        'live_experiment_view.py', 'trial_assistance.py')]
    paths += sorted((ROOT / 'src/fdm/services').glob('optical_*.py'))
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('quick', 'full'), default='quick')
    parser.add_argument('--algorithm', choices=('traditional', 'multiscale', 'optical_adaptive'), default='traditional')
    parser.add_argument('--scenario', choices=('quick', 'cycle', 'clicks', 'menus', 'latency', 'retention', 'performance'), default='quick')
    parser.add_argument('--performance-mode', choices=('baseline', 'candidates'), default='candidates')
    parser.add_argument('--duration', type=float)
    parser.add_argument('--clicks', type=int)
    parser.add_argument('--delay-kind', choices=('gui', 'motion'), default='gui')
    parser.add_argument('--delay-ms', type=int, choices=(0, 40, 80, 200), default=0)
    parser.add_argument('--theme', choices=('light', 'dark', 'navy', 'warm'), default='light')
    parser.add_argument('--size', default='1600x900')
    parser.add_argument('--scale', type=float, default=1.)
    parser.add_argument('--base-device-scale', type=float, default=1.)
    parser.add_argument('--platform', default='windows' if sys.platform == 'win32' else 'offscreen')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.duration = args.duration if args.duration is not None else (3 if args.phase == 'quick' else 60)
    args.clicks = args.clicks if args.clicks is not None else (3 if args.phase == 'quick' else 100)
    if args.duration <= 0 or args.clicks <= 0:
        parser.error('duration and clicks must be positive')
    return args


def main(args):
    os.environ['QT_QPA_PLATFORM'] = args.platform
    os.environ['QT_SCALE_FACTOR'] = str(args.scale / args.base_device_scale)
    sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts/validation')]
    from PySide6.QtCore import QEventLoop, QPoint, Qt, QTimer
    from PySide6.QtGui import QImage
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QDialogButtonBox, QPushButton, QTableWidget
    from validate_workflow_redesign import WorkflowHarness, dimensions, synthetic_frame
    from fdm.live_experiment.candidate_alignment import CandidateAlignmentMonitor
    from fdm.live_experiment.candidate_alignment_worker import AlignmentProcessWorker
    from fdm.live_experiment.controller import LiveExperimentController
    from fdm.live_experiment.frames import FrameHandle
    from fdm.live_experiment.motion import FieldMotionDetector
    from fdm.ui.live_measurement_canvas import LiveMeasurementCanvas

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'result.json').exists():
        raise FileExistsError('Preserve earlier evidence; choose a new output directory')
    (output / 'validation-script.py').write_bytes(Path(__file__).read_bytes())
    profile = Path(tempfile.mkdtemp(prefix='profile-', dir=output))
    physical = dimensions(args.size)
    logical = tuple(round(value / args.scale) for value in physical)
    started = time.perf_counter()
    stage = dict(name='setup', inject=False)
    trace = dict(analyses=[], visibility=[], status=[], actions=[], delayed=[], field_changes=[], retention=[])
    report = dict(schema_version=1, started_at=datetime.now(timezone.utc).isoformat(),
        phase=args.phase, scenario=args.scenario, runtime=platform.platform(), platform=args.platform,
        algorithm=args.algorithm,
        theme=args.theme, scale=args.scale, requested_client_raster=physical,
        source_fingerprints=fingerprints(), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        real_camera=False, passed=False, trace=trace,
        scope='Actual navigation, analyzer, global field detector, pointer clicks, ledger and evidence; synthetic camera.')
    alignment_calls = dict(poll=0, compare=0)
    h = page = c = ticker = None
    recording = dict(process=None, timer=None, frames=0, errors=[])
    performance_samples = dict(active=False, frames=[], previous_frame=None)

    def now():
        return round(time.perf_counter() - started, 6)

    def pause(ms):
        loop, timer = QEventLoop(), QTimer()
        timer.setTimerType(Qt.TimerType.PreciseTimer)
        timer.setSingleShot(True)
        timer.timeout.connect(loop.quit)
        timer.start(max(1, round(ms)))
        loop.exec()

    def wait(predicate, label, timeout=15):
        deadline = time.perf_counter() + timeout
        while not predicate() and time.perf_counter() < deadline:
            pause(10)
        if not predicate():
            raise AssertionError(dict(label=label, stage=stage['name'], state=c.state if c else None,
                status=page.statusBar().currentMessage() if page else None,
                accepted=len(page.session.measurements) if page else None,
                recent_visibility=trace['visibility'][-8:]))

    def marks():
        return list(c.selection_retention.visible_marks())

    def mark_dict(value):
        return asdict(value) if is_dataclass(value) else dict(value)

    original_paint = LiveMeasurementCanvas.paintEvent
    original_motion = LiveExperimentController._prepared_motion_work
    original_direct_motion = LiveExperimentController._motion_work
    original_presented = LiveExperimentController.mark_presented
    original_alignment_poll = CandidateAlignmentMonitor.poll
    original_alignment_compare = AlignmentProcessWorker.compare

    def painted(canvas, event):
        original_paint(canvas, event)
        if page is None or canvas is not page.canvas:
            return
        context = c._automatic_context
        row = dict(visible=canvas._candidates_visible(), batch=context.get('batch_id') if context else None,
            available=sum(item.get('state') in ('available', 'confirming') for item in c.candidates),
            stage=stage['name'])
        previous = trace['visibility'][-1] if trace['visibility'] else None
        if previous is None or any(previous[key] != value for key, value in row.items()):
            trace['visibility'].append(dict(at=now(), **row))

    def delayed_motion(controller, *values, **kwargs):
        result = original_motion(controller, *values, **kwargs)
        return delay_motion_result(result, 'prepared')

    def delayed_direct_motion(controller, *values, **kwargs):
        result = original_direct_motion(controller, *values, **kwargs)
        return delay_motion_result(result, 'direct')

    def delay_motion_result(result, pipeline):
        if stage['inject'] and args.delay_kind == 'motion' and args.delay_ms:
            began = time.perf_counter()
            time.sleep(args.delay_ms / 1000)
            trace['delayed'].append(dict(at=now(), kind='global-motion', pipeline=pipeline,
                                        elapsed_ms=(time.perf_counter()-began)*1000))
        return result

    def observed_presented(controller, token):
        result = original_presented(controller, token)
        if (controller is c and performance_samples['active'] and controller.presented is token
                and token.frame.frame_id != performance_samples['previous_frame']):
            performance_samples['previous_frame'] = token.frame.frame_id
            sampled = time.perf_counter()
            performance_samples['frames'].append(dict(at=sampled, age_ms=(sampled-token.frame.received_at)*1000))
        return result

    def alignment_poll(monitor, *values, **kwargs):
        alignment_calls['poll'] += 1
        return original_alignment_poll(monitor, *values, **kwargs)

    def alignment_compare(worker, *values, **kwargs):
        alignment_calls['compare'] += 1
        return original_alignment_compare(worker, *values, **kwargs)

    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(LiveMeasurementCanvas, 'paintEvent', painted))
            stack.enter_context(patch.object(LiveExperimentController, '_prepared_motion_work', delayed_motion))
            stack.enter_context(patch.object(LiveExperimentController, '_motion_work', delayed_direct_motion))
            stack.enter_context(patch.object(LiveExperimentController, 'mark_presented', observed_presented))
            stack.enter_context(patch.object(CandidateAlignmentMonitor, 'poll', alignment_poll))
            stack.enter_context(patch.object(AlignmentProcessWorker, 'compare', alignment_compare))
            h = stack.enter_context(WorkflowHarness(profile, theme=args.theme, size=logical))
            h.enter()
            page = h.select(1, frame=False)
            c = page.controller
            source = dict(image=synthetic_frame(sequence=0))
            ticker = QTimer(h.window)
            ticker.setTimerType(Qt.TimerType.PreciseTimer)
            ticker.setInterval(33)
            ticker.timeout.connect(lambda: h.backend.tick(image=source['image']))
            ticker.start()
            c.statusChanged.connect(lambda message: trace['status'].append(dict(at=now(), message=message)))
            c.fieldChanged.connect(lambda *values: trace['field_changes'].append(dict(at=now(), values=values)))
            original_analyze = c.analyze
            def observed_analyze(*values, **kwargs):
                result = original_analyze(*values, **kwargs)
                context = c._automatic_context
                trace['analyses'].append(dict(at=now(), origin=kwargs.get('_origin', 'manual'),
                    batch=context.get('batch_id') if context else None, stage=stage['name']))
                return result
            c.analyze = observed_analyze

            def rename():
                dialog = QApplication.activeModalWidget()
                dialog.setTextValue('Round 7 Operator')
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
            wait(lambda: c.state == 'ready' and page.analyze_button.isEnabled(), 'prepared')
            assert page.input_mode.currentData() == 'mixed'
            assert not page.auto_analysis.isChecked()
            if args.algorithm != page.algorithm.currentData():
                menu = page.analyze_button.menu()
                menu.popup(page.analyze_button.mapToGlobal(page.analyze_button.rect().bottomLeft()))
                pause(40)
                action = page.algorithm_actions[args.algorithm]
                QTest.mouseClick(menu, Qt.MouseButton.LeftButton, pos=menu.actionGeometry(action).center())
                assert page.algorithm.currentData() == args.algorithm
            baseline = args.scenario == 'performance' and args.performance_mode == 'baseline'
            if not baseline:
                QTest.mouseClick(page.auto_analysis, Qt.MouseButton.LeftButton,
                                 pos=QPoint(8, page.auto_analysis.height() // 2))
                wait(lambda: bool(c.candidates) and page.canvas._candidates_visible(), 'initial automatic analysis')
            else:
                page.measure_mode.setCurrentIndex(page.measure_mode.findData('free'))
            batch = c._automatic_context['batch_id'] if c._automatic_context else None
            initial_ids = [item['candidate_id'] for item in c.candidates]
            report['initial_candidates'] = len(initial_ids)
            initial_capture = h.window.grab()
            raster = (initial_capture.width(), initial_capture.height())
            report['actual_client_raster'] = raster
            initial_capture.save(str(output / 'initial.png'))
            del initial_capture
            pause(100)

            if args.video:
                ffmpeg = shutil.which('ffmpeg')
                if not ffmpeg:
                    raise RuntimeError('--video requires ffmpeg')
                recording['process'] = subprocess.Popen([ffmpeg, '-y', '-loglevel', 'error', '-f', 'rawvideo',
                    '-pixel_format', 'rgba', '-video_size', f'{raster[0]}x{raster[1]}', '-framerate', '8', '-i', '-',
                    '-an', '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '24', '-pix_fmt', 'yuv420p',
                    str(output / 'workflow.mp4')], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
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
                pause(250)  # Capture the actual initial state before pointer actions.

            new_field = next(button for button in page.findChildren(QPushButton) if button.text() == '新视场')

            def painted_available_batch(previous=None):
                context = c._automatic_context
                if (not context or not context.get('completed')
                        or context['batch_id'] == previous or not page.canvas._candidates_visible()):
                    return False
                available = {item['candidate_id'] for item in c.candidates if item.get('state') == 'available'}
                painted = {item['candidate_id'] for item in page.canvas._painted_candidates}
                return bool(available and available <= painted)

            def next_field():
                previous = c._automatic_context['batch_id'] if c._automatic_context else None
                QTest.mouseClick(new_field, Qt.MouseButton.LeftButton)
                wait(lambda: painted_available_batch(previous), 'next field analysis completed and painted')

            def real_motion():
                import numpy as np
                rgba = source['image'].convertToFormat(QImage.Format.Format_RGBA8888)
                pixels = np.frombuffer(rgba.constBits(), np.uint8).reshape(rgba.height(), rgba.width(), 4).copy()
                moved = pixels[:, ::-1].copy()
                # Certify that the chosen stimulus really exercises the
                # unchanged detector, rather than replacing its decision.
                detector = FieldMotionDetector()
                detector.compare(FrameHandle.snapshot(pixels, source_id='fixture', sequence=1))
                result = detector.compare(FrameHandle.snapshot(moved, source_id='fixture', sequence=2))
                assert not result.stable, 'The movement fixture must change the actual field'
                previous_epoch = c.field_epoch
                source['image'] = QImage(moved.data, moved.shape[1], moved.shape[0], moved.strides[0],
                                          QImage.Format.Format_RGBA8888).copy()
                wait(lambda: c.field_epoch > previous_epoch, 'actual source motion ends field')
                return dict(previous_epoch=previous_epoch, changed_epoch=c.field_epoch,
                            detector_reason=result.reason, at=now())

            def direct_click_target():
                # The optical pipeline deliberately preserves mutually exclusive
                # interpretations. This exercise measures direct acceptance, so
                # choose a visible, uniquely hit proposal that does not require
                # the existing duplicate review. Separate tests cover that review.
                scale, offset = page.canvas._transform()
                for index, value in enumerate(c.candidates):
                    if value.get('state') != 'available' or c.possible_duplicate_candidates([index]):
                        continue
                    center = tuple((a+b)/2 for a, b in zip(value['start'], value['end']))
                    point = QPoint(round(center[0]*scale+offset[0]), round(center[1]*scale+offset[1]))
                    if page.canvas.candidate_hits(point) == [value['candidate_id']]:
                        return value, point
                return None

            def click_one():
                target = direct_click_target()
                if target is None:
                    next_field()
                    target = direct_click_target()
                assert target is not None, 'At least one unambiguous direct-click target is required'
                item, point = target
                ids = [value['candidate_id'] for value in c.candidates]
                current_batch = c._automatic_context['batch_id']
                page.image_scroll.ensureVisible(point.x(), point.y(), 20, 20)
                wait(lambda: page.canvas._candidates_visible() and c.presented is not None
                     and c.presented.frame is page.canvas.frame, 'actually painted click')
                before = set(page.session.measurements)
                began = time.perf_counter()
                QTest.mouseClick(page.canvas, Qt.MouseButton.LeftButton, pos=point)
                wait(lambda: len(page.session.measurements) == len(before)+1, 'point selection accepted')
                key = (set(page.session.measurements)-before).pop()
                trace['actions'].append(dict(at=now(), candidate=item['candidate_id'], record=key,
                    batch=current_batch, elapsed_ms=(time.perf_counter()-began)*1000))
                assert [value['candidate_id'] for value in c.candidates] == ids
                assert c._automatic_context['batch_id'] == current_batch
                assert page.workflow_tabs.currentIndex() == 0
                return key

            def manual_one(index):
                scale, offset = page.canvas._transform()
                start, end = (100 + (index % 20)*5, 240), (120 + (index % 20)*5, 240)
                first = QPoint(round(start[0]*scale+offset[0]), round(start[1]*scale+offset[1]))
                second = QPoint(round(end[0]*scale+offset[0]), round(end[1]*scale+offset[1]))
                page.image_scroll.ensureVisible(first.x(), first.y(), 20, 20)
                wait(lambda: c.presented is not None and c.presented.frame is page.canvas.frame,
                     'manual measurement painted frame')
                count = len(page.session.measurements)
                began = time.perf_counter()
                QTest.mouseClick(page.canvas, Qt.MouseButton.LeftButton, pos=first)
                assert c.draft is not None
                QTest.mouseClick(page.canvas, Qt.MouseButton.LeftButton, pos=second)
                wait(lambda: len(page.session.measurements) == count+1, 'manual baseline accepted')
                trace['actions'].append(dict(at=now(), kind='manual-free', elapsed_ms=(time.perf_counter()-began)*1000))

            if args.scenario in ('quick', 'cycle'):
                stage['name'] = 'static'
                began = time.perf_counter()
                while time.perf_counter()-began < args.duration:
                    pause(min(500, (args.duration-(time.perf_counter()-began))*1000+1))
                assert len(trace['analyses']) == 1
                assert c._automatic_context['batch_id'] == batch
                assert [item['candidate_id'] for item in c.candidates] == initial_ids
                report['static'] = dict(seconds=time.perf_counter()-began, analyses=1, source_frames=h.backend.sequence)

            if args.scenario in ('quick', 'clicks', 'retention'):
                stage['name'] = 'clicks'
                accepted = []
                for index in range(args.clicks):
                    if not any(item.get('state') == 'available' for item in c.candidates):
                        next_field()
                    accepted.append(click_one())
                    pause(60)
                wait(lambda: all(value.evidence.get('state') == 'saved' for value in page.session.measurements.values()),
                     'all evidence saved')
                report['clicks'] = dict(accepted=len(accepted), records=len(page.session.measurements),
                    saved=sum(value.evidence.get('state') == 'saved' for value in page.session.measurements.values()))

            if args.scenario in ('quick', 'retention'):
                stage['name'] = 'retention'
                selected = set(c.current_records)
                assert selected
                assert not marks(), 'Retention must begin on field transition only'
                previous_batch = c._automatic_context['batch_id']
                began = time.perf_counter()
                QTest.mouseClick(new_field, Qt.MouseButton.LeftButton)
                current = [mark_dict(value) for value in marks()]
                assert {value['record_id'] for value in current} == selected
                report['retention'] = dict(records=sorted(selected), marks=current)
                h.window.grab().save(str(output / 'retained-markers.png'))
                # Ordinary menu and new-field requests must not extend existing
                # deadlines. The second field has no newly selected record.
                pause(120)
                QTest.mouseClick(new_field, Qt.MouseButton.LeftButton)
                assert {value['record_id'] for value in map(mark_dict, marks())} == selected
                wait(lambda: not marks(), 'retention expires without renewal', timeout=2)
                report['retention']['elapsed_seconds'] = time.perf_counter()-began
                assert report['retention']['elapsed_seconds'] < 1.25
                wait(lambda: painted_available_batch(previous_batch), 'new field after retention')

            if args.scenario == 'cycle':
                # A gradual focus-like change can remain in the same business
                # field. It must not create a second candidate gating policy.
                import cv2
                import numpy as np
                from fdm.live_experiment.motion import reduced_gray, sobel_fraction
                rgba = source['image'].convertToFormat(QImage.Format.Format_RGBA8888)
                pixels = np.frombuffer(rgba.constBits(), np.uint8).reshape(rgba.height(), rgba.width(), 4).copy()
                epoch = c.field_epoch
                stage['name'] = 'stable-focus-change'
                previous = pixels
                ramp = []
                for sigma in np.arange(.1, 1.21, .1):
                    blurred = cv2.GaussianBlur(pixels, (0, 0), float(sigma))
                    fraction = sobel_fraction(reduced_gray(previous), reduced_gray(blurred))
                    assert fraction <= .04138815884775365
                    ramp.append(dict(sigma=float(sigma), changed_fraction=fraction))
                    source['image'] = QImage(blurred.data, blurred.shape[1], blurred.shape[0], blurred.strides[0],
                                              QImage.Format.Format_RGBA8888).copy()
                    pause(80)
                    assert c.field_epoch == epoch and c.state == 'ready'
                    assert c._automatic_context['batch_id'] == batch
                    assert page.canvas._candidates_visible()
                    previous = blurred
                assert len(trace['analyses']) == 1
                report['stable_focus_change'] = dict(ramp=ramp, batch_preserved=True, analyses=1)
                selected_before_motion = {click_one() for _ in range(3)}
                before = len(trace['analyses'])
                stage['name'] = 'actual-motion'
                report['actual_motion'] = real_motion()
                retained = {value['record_id'] for value in map(mark_dict, marks())}
                assert retained == selected_before_motion
                report['actual_motion']['retained_ids'] = sorted(retained)
                h.window.grab().save(str(output / 'motion-retained-markers.png'))
                wait(lambda: painted_available_batch(batch), 'actual motion produces new candidates')
                pause(500)
                assert len(trace['analyses']) == before+1
                wait(lambda: not marks(), 'motion retained markers expire', timeout=2)
                report['cycle'] = dict(actual_motion_additional_analyses=1)

            if args.scenario == 'latency':
                stage.update(name='latency', inject=True)
                for _ in range(6):
                    if args.delay_kind == 'gui':
                        began = time.perf_counter()
                        time.sleep(args.delay_ms/1000)
                        trace['delayed'].append(dict(at=now(), kind='gui', elapsed_ms=(time.perf_counter()-began)*1000))
                    pause(150)
                stage['inject'] = False
                pause(500)
                assert c._automatic_context['batch_id'] == batch
                assert len(trace['analyses']) == 1
                assert page.canvas._candidates_visible()
                assert args.delay_ms == 0 or len(trace['delayed']) >= 3, 'Requested delay was not exercised'
                report['latency'] = dict(kind=args.delay_kind, delay_ms=args.delay_ms, batch_preserved=True)

            if args.scenario == 'menus':
                stage['name'] = 'menus'
                opened, menu_errors = [], []
                trace['menu_events'], trace['menu_errors'] = opened, menu_errors
                for index in range(100):
                    # QToolButton InstantPopup enters a nested event loop;
                    # dismiss using an actual Escape key in that same loop.
                    control = page.candidate_display_button if index % 2 == 0 else page.measure_mode
                    expected = 'candidate-display' if index % 2 == 0 else 'manual-algorithm'
                    def dismiss(name=expected):
                        popup = QApplication.activePopupWidget()
                        if popup is None or not popup.isVisible():
                            menu_errors.append(dict(menu=name, error='No visible popup after real pointer click'))
                        else:
                            opened.append(dict(at=now(), menu=name, popup_type=type(popup).__name__))
                            key_target = page.measure_mode.view() if name == 'manual-algorithm' else popup
                            QTest.keyClick(key_target, Qt.Key.Key_Escape)
                    QTimer.singleShot(60, dismiss)
                    QTest.mouseClick(control, Qt.MouseButton.LeftButton,
                                     pos=QPoint(control.width()-6, control.height()//2))
                    pause(80)
                    assert QApplication.activePopupWidget() is None
                    assert c._automatic_context['batch_id'] == batch
                    assert [item['candidate_id'] for item in c.candidates] == initial_ids
                    assert page.canvas._candidates_visible()
                assert not menu_errors and len(opened) == 100, menu_errors
                assert len(trace['analyses']) == 1
                report['menus'] = dict(opened=opened, count=len(opened), batch_preserved=True, analyses=1)

            if args.scenario == 'performance':
                # Both runs use the same source, 33 ms producer, JPEG policy
                # and one successful measurement per second. This is a
                # current-build workload comparison, not a previous-build
                # speed claim. Startup metrics are kept separate.
                import numpy as np
                report['startup_performance'] = c.performance_diagnostics()
                pause(1000)
                stage['name'] = 'performance'
                performance_samples['active'] = True
                began = time.perf_counter()
                next_measure = began+1
                measured = 0
                while time.perf_counter()-began < args.duration:
                    if time.perf_counter() >= next_measure:
                        if baseline:
                            manual_one(measured)
                        else:
                            if not any(value.get('state') == 'available' for value in c.candidates):
                                next_field()
                            click_one()
                        measured += 1
                        next_measure += 1
                    pause(5)
                performance_samples['active'] = False
                elapsed = time.perf_counter()-began
                rows = performance_samples['frames']
                assert len(rows) >= 10
                # Repeated new-field analysis is part of the assisted work;
                # initial process startup is outside this timed load window.
                report['performance_load'] = dict(mode=args.performance_mode, seconds=elapsed,
                    sample_frames=len(rows), fps=(len(rows)-1)/(rows[-1]['at']-rows[0]['at']),
                    p50_frame_age_ms=float(np.quantile([row['age_ms'] for row in rows], .5)),
                    p95_frame_age_ms=float(np.quantile([row['age_ms'] for row in rows], .95)),
                    measurements=measured, measurement_period_ms=1000, source_period_ms=33,
                    evidence_options=page.session.effective_evidence_options,
                    comparison_scope='Current build baseline manual-free vs automatic candidate workflow; synthetic camera.')
                wait(lambda: all(value.evidence.get('state') == 'saved' for value in page.session.measurements.values()),
                     'performance evidence complete')
                report['performance_load']['saved'] = sum(value.evidence.get('state') == 'saved'
                                                         for value in page.session.measurements.values())

            hides = [new for previous, new in zip(trace['visibility'], trace['visibility'][1:])
                     if previous['visible'] and not new['visible'] and previous['batch'] == new['batch']
                     and new['available'] and new['stage'] != 'setup']
            report['remaining_candidate_hides'] = hides
            assert not hides, hides
            assert alignment_calls == dict(poll=0, compare=0), alignment_calls
            assert not h.slot_errors, h.slot_errors
            assert not h.modal_messages, h.modal_messages
            report['performance'] = c.performance_diagnostics()
            report['context'] = c.performance_frame_context()
            report['accepted_total'] = len(page.session.measurements)
            report['passed'] = True
            if recording['timer'] is not None:
                pause(300)  # Include the observed expired state, not only its assertion.
            h.window.grab().save(str(output / 'final.png'))
            ticker.stop()
            if recording['timer'] is not None:
                recording['timer'].stop()
    except Exception:
        report['error'] = traceback.format_exc()
        if h is not None:
            report['slot_errors'], report['modal_messages'] = h.slot_errors, h.modal_messages
        if c is not None:
            report['failure_context'] = c.performance_frame_context()
    finally:
        if recording['process'] is not None:
            process = recording['process']
            process.stdin.close()
            code = process.wait(timeout=30)
            report['video'] = dict(frames=recording['frames'], fps=8, kind='Qt grabs; diagnostic overhead',
                exit_code=code, errors=recording['errors'], stderr=process.stderr.read().decode(errors='replace'))
            if code or recording['errors']:
                report['passed'] = False
        report['independent_alignment_calls'] = alignment_calls
        report['source_changed_during_run'] = fingerprints() != report['source_fingerprints']
        report['elapsed_seconds'] = now()
        (output / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'], output=str(output), error=report.get('error')), ensure_ascii=False))
    return int(not report['passed'])


if __name__ == '__main__':
    raise SystemExit(main(arguments()))
