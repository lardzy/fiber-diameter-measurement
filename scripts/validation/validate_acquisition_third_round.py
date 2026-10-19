"""Third-round real-navigation visual checks with an isolated synthetic camera.

The video is a sequence of Qt widget grabs, including a composited floating dock;
it is not a desktop recording or evidence of native camera/window stacking.
"""
from __future__ import annotations
import argparse
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

ROOT = Path(__file__).resolve().parents[2]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--theme', choices=('light', 'dark', 'navy', 'warm'), default='light')
    parser.add_argument('--size', default='1600x900', help='Physical Qt client raster, excludes OS decorations/taskbar')
    parser.add_argument('--scale', type=float, choices=(1, 1.25, 1.5, 2), default=1)
    parser.add_argument('--platform', default='windows' if sys.platform == 'win32' else 'offscreen')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cycles', type=int, default=200)
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--matrix', action='store_true', help='Run four themes across nine client/DPI cases')
    parser.add_argument('--workers', type=int, default=2)
    return parser.parse_args()


def fingerprints():
    names = ('src/fdm/ui/live_experiment_view.py', 'src/fdm/ui/live_experiment_workspace.py',
             'src/fdm/ui/quantity_targets.py', 'src/fdm/ui/trial_records.py',
             'src/fdm/ui/trial_report_panel.py', 'src/fdm/ui/acquisition_workbench.py',
             'src/fdm/ui/theme.py', 'src/fdm/ui/digital_slide_canvas.py', 'src/fdm/ui/dialogs.py')
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def main(args):
    os.environ['QT_QPA_PLATFORM'] = args.platform
    os.environ['QT_SCALE_FACTOR'] = str(args.scale)
    sys.path.insert(0, str(ROOT / 'src'))
    from PySide6.QtCore import QPoint, Qt, QTimer
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtTest import QTest
    from validate_workflow_redesign import WorkflowHarness, dimensions, settle
    from fdm.live_experiment import built_in_materials
    from fdm.live_experiment.feedback import feedback_for_reason

    physical = dimensions(args.size)
    logical = tuple(round(value / args.scale) for value in physical)
    key = f'{args.theme}-{args.size}-{format(args.scale, 'g').replace(".", "p")}'
    output = (args.output or ROOT / '.tmp/acquisition-third-round-20261019' / key).resolve()
    output.mkdir(parents=True, exist_ok=True)
    profile = Path(tempfile.mkdtemp(prefix='profile-', dir=output))
    report = dict(schema_version=1, created_at=datetime.now(timezone.utc).isoformat(),
                  theme=args.theme, physical_client=physical, logical_client=logical, scale=args.scale,
                  runtime=platform.platform(), platform=args.platform, real_hardware=False,
                  validation_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  real_windows=sys.platform == 'win32', source_fingerprints=fingerprints(),
                  screenshots=[], transitions=[], records={}, passed=False,
                  limitations=['Real MainWindow navigation, synthetic camera, isolated application preferences.',
                               'Record fixtures use domain mutations; candidates enter at the controller presentation signal.',
                               'Qt scale factors do not validate physical monitor migration or native OS stacking.',
                               'Video uses Qt widget grabs and composites the application-owned floating dock; not a desktop recording.',
                               'Excel report visibility only; this visual pass does not invoke Excel COM.'])
    started = time.perf_counter()
    recorder = dict(process=None, timer=None, frames=0, timeline=[], errors=[])
    h = None
    try:
        with WorkflowHarness(profile, theme=args.theme, size=logical) as h:
            w = h.window
            h.enter()
            records = h.hub.records
            report['font'] = h.font_report()
            report['dpr'] = w.devicePixelRatioF()

            def rect(widget):
                point = widget.mapTo(w, QPoint())
                return [point.x(), point.y(), widget.width(), widget.height()]

            navigation = rect(h.hub.navigation)
            def capture():
                result = w.grab()
                if records.dock.isVisible() and records.dock.isFloating():
                    painter = QPainter(result)
                    offset = records.dock.mapToGlobal(QPoint()) - w.mapToGlobal(QPoint())
                    painter.drawPixmap(offset, records.dock.grab())
                    painter.end()
                return result

            if args.video:
                ffmpeg = shutil.which('ffmpeg')
                if not ffmpeg:
                    raise RuntimeError('ffmpeg is required for --video')
                raster = w.grab().size()
                recorder['process'] = subprocess.Popen([ffmpeg, '-y', '-loglevel', 'error', '-f', 'rawvideo',
                    '-pixel_format', 'rgba', '-video_size', f'{raster.width()}x{raster.height()}', '-framerate', '8',
                    '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '24', '-pix_fmt', 'yuv420p',
                    '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', str(output / 'navigation-records.mp4')],
                    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                recorder['timer'] = QTimer(w)
                recorder['timer'].setInterval(125)
                def video_frame():
                    try:
                        image = capture().toImage().convertToFormat(QImage.Format.Format_RGBA8888)
                        assert image.size() == raster, 'window raster changed during video'
                        recorder['process'].stdin.write(image.constBits())
                        recorder['frames'] += 1
                    except Exception as exc:
                        recorder['errors'].append(str(exc))
                        recorder['timer'].stop()
                recorder['timer'].timeout.connect(video_frame)
                recorder['timer'].start()

            def snap(name, page=None):
                QTest.qWait(180 if args.video else 100)
                settle()
                h.assert_size()
                assert rect(h.hub.navigation) == navigation, (name, navigation, rect(h.hub.navigation))
                assert h.hub.navigation.isVisible()
                pixmap = capture()
                path = output / (name + '.png')
                assert pixmap.save(str(path))
                row = dict(name=name, path=str(path), raster=[pixmap.width(), pixmap.height()],
                           sha256=hashlib.sha256(path.read_bytes()).hexdigest(), navigation=rect(h.hub.navigation),
                           records_open=records.is_open, record_position=records.preferences['position'])
                if page is not None:
                    row.update(image=rect(page.image_scroll), export_button=rect(page.export_button),
                               export_text=page.export_button.text(), category_size=[page.category_scroll.width(), page.category_scroll.height()])
                    assert page.export_button.isVisible()
                    assert page.trial_sidebar.rect().contains(page.export_button.mapTo(page.trial_sidebar, page.export_button.rect().center()))
                report['screenshots'].append(row)
                recorder['timeline'].append(dict(name=name, frame=recorder['frames'], elapsed=round(time.perf_counter() - started, 3)))
                return row

            def tab(page, index):
                bar = page.workflow_tabs.tabBar()
                QTest.mouseClick(bar, Qt.MouseButton.LeftButton, pos=bar.tabRect(index).center())
                settle()
                assert page.workflow_tabs.currentIndex() == index

            h.select(0)
            snap('realtime')
            page = h.select(1)
            assert not page.material_rows and page.empty_categories.isVisible()
            assert not w.project.live_experiments
            snap('content-empty', page)
            # Domain fixtures configure categories, never bypass workspace navigation.
            session = page.session
            materials = built_in_materials('normal')[:9]
            session.rename_operator('张三')
            session.set_materials(materials, direction='warp')
            session.set_materials(materials, direction='weft')
            page._configure(sample_name='含量试验 · 长样品名称视觉回归', sample_number='000123')
            page._refresh_operators()
            page.ruler_choice.setCurrentIndex(1)
            page.ruler_choice.activated.emit(1)
            page._changed()
            page._refresh_summary()
            settle()
            vp = page.category_scroll.viewport()
            visible = []
            for row in page.material_rows.values():
                control = row['select']
                point = control.mapTo(vp, QPoint())
                if vp.rect().contains(point) and vp.rect().contains(point + control.rect().bottomRight()):
                    visible.append(control.text())
            report['visible_categories'] = visible
            minimum = 6 if logical[1] >= 900 else 4 if logical[1] >= 720 else 1
            assert len(visible) >= minimum, (logical, len(visible), minimum)
            snap('content-nine', page)
            QTest.mouseClick(page.sample_details_toggle, Qt.MouseButton.LeftButton)
            assert page.sample_details.isVisible()
            snap('content-metadata', page)
            QTest.mouseClick(page.sample_details_toggle, Qt.MouseButton.LeftButton)
            settle()
            # Three operators and signed count ledger events, across both directions.
            owners = [session.active_operator_id, session.add_operator('李四'), session.add_operator('王五')]
            for owner in owners:
                for direction in ('warp', 'weft'):
                    for index in range(28):
                        session.add_measurement(18 + index / 10, materials[index % 9].id,
                            operator_id=owner, direction=direction, field_id=f'{owner}-{direction}')
                    session.adjust_count(materials[0].id, 3, operator_id=owner, direction=direction, note='视觉回归：补计')
                    session.adjust_count(materials[0].id, -1, operator_id=owner, direction=direction, note='视觉回归：修正补计')
            session.active_operator_id = owners[0]
            page._refresh_operators()
            page._changed()
            page._refresh_summary()
            QTest.mouseClick(page.records_toggle, Qt.MouseButton.LeftButton)
            settle()
            panel = records.panel
            assert records.is_open
            assert panel.operator_filter.currentData() is None and panel.direction_filter.currentData() is None
            assert panel.model.rowCount() == 180 and panel.proxy.rowCount() == 180
            def position(label):
                action = next(action for action in panel.position_button.menu().actions() if action.text() == label)
                action.trigger()
                settle()
                QTest.qWait(100)
                if panel.following:
                    scrollbar = panel.table.verticalScrollBar()
                    assert scrollbar.value() == scrollbar.maximum(), (label, scrollbar.value(), scrollbar.maximum())
            position('停靠左侧')
            snap('records-left', page)
            scrollbar = panel.table.verticalScrollBar()
            assert scrollbar.value() == scrollbar.maximum()
            scrollbar.setFocus()
            QTest.keyClick(scrollbar, Qt.Key.Key_Home)
            settle()
            before = scrollbar.value()
            assert not panel.following
            new_record = session.add_measurement(22, materials[0].id, operator_id=owners[2], direction='weft')
            page.controller.recordAccepted.emit(new_record)
            page.controller.changed.emit()
            settle()
            assert scrollbar.value() == before and not panel.following
            assert '新增 1 条' in panel.latest_button.text()
            report['records'].update(total_initial=180, operators=3, signed_count_events=12,
                scrolled_up_position=before, position_after_append=scrollbar.value(), new_message=panel.latest_button.text())
            snap('records-new-while-reading', page)
            QTest.mouseClick(panel.latest_button, Qt.MouseButton.LeftButton)
            settle()
            assert scrollbar.value() == scrollbar.maximum() and panel.following
            snap('records-latest', page)
            position('停靠底部')
            snap('records-bottom', page)
            QTest.mouseClick(page.sample_details_toggle, Qt.MouseButton.LeftButton)
            QTest.qWait(250)
            snap('records-bottom-metadata', page)
            page.metadata_scroll.verticalScrollBar().setFocus()
            QTest.keyClick(page.metadata_scroll.verticalScrollBar(), Qt.Key.Key_Home)
            settle()
            QTest.mouseClick(page.sample_details_toggle, Qt.MouseButton.LeftButton)
            QTest.qWait(250)
            position('浮动窗口')
            snap('records-floating', page)
            floating = output / 'records-floating-window.png'
            records.dock.grab().save(str(floating))
            report['floating_window_screenshot'] = str(floating)
            assert records.dock.isFloating()
            QTest.mouseClick(page.records_toggle, Qt.MouseButton.LeftButton)
            settle()
            assert not records.is_open
            tab(page, 2)
            snap('content-report', page)
            tab(page, 1)
            page.controller.candidatesChanged.emit([{'width_px': 36}, {'width_px': 48}])
            snap('content-candidates', page)
            page.controller.candidatesChanged.emit([])
            tab(page, 0)
            page.controller.measurementFeedback.emit(feedback_for_reason('no_complete_crossing', **page.controller.feedback_context()))
            snap('content-rejected', page)
            page.measurement_feedback.dismiss()
            page = h.select(2)
            snap('diameter', page)
            for index in range(args.cycles):
                selected = (0, 1, 2)[index % 3]
                trial = h.select(selected)
                assert rect(h.hub.navigation) == navigation
                assert not records.is_open
                if index < 3 or index >= args.cycles - 3:
                    report['transitions'].append(dict(step=index, page=selected, navigation=rect(h.hub.navigation)))
                if args.video and index < 12:
                    QTest.qWait(140)
            report['navigation_cycles'] = args.cycles
            h.select(0)
            snap('realtime-return')
            assert not h.slot_errors, h.slot_errors
            assert not h.modal_messages, h.modal_messages
            unexpected = [event for event in h.observer.events if event['class'] not in ('MainWindow', 'QTipLabel', '_RecordsDock', 'QMenu')]
            assert not unexpected, unexpected
            report['window_events'] = h.observer.events
            report['passed'] = True
            if recorder['timer'] is not None:
                QTest.qWait(250)
                recorder['timer'].stop()
    except Exception:
        report['error'] = traceback.format_exc()
        if h is not None:
            report['slot_errors'] = h.slot_errors
    finally:
        if recorder['process'] is not None:
            process = recorder['process']
            process.stdin.close()
            code = process.wait(timeout=30)
            errors = process.stderr.read().decode('utf-8', errors='replace')
            report['video'] = dict(path=str(output / 'navigation-records.mp4'), fps=8, frames=recorder['frames'],
                duration=recorder['frames'] / 8, kind='Qt widget grab sequence with floating dock compositing',
                timeline=recorder['timeline'], errors=recorder['errors'], ffmpeg_exit=code, ffmpeg_stderr=errors)
            if code or recorder['errors']:
                report['passed'] = False
        report['source_changed_during_run'] = fingerprints() != report['source_fingerprints']
        report['elapsed_seconds'] = round(time.perf_counter() - started, 3)
        (output / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'], result=str(output / 'result.json'), error=report.get('error')), ensure_ascii=False))
    return 0 if report['passed'] else 1


def matrix(args):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    base = (args.output or ROOT / '.tmp/acquisition-third-round-20261019').resolve()
    base.mkdir(parents=True, exist_ok=True)
    cases = [(theme, size, scale) for theme in ('light', 'dark', 'navy', 'warm')
             for size, scale in [('1600x900', 1), ('1600x900', 1.25), ('1600x900', 1.5),
                 ('1366x768', 1), ('1920x1080', 1), ('1920x1080', 1.25), ('1920x1080', 1.5),
                 ('3840x2160', 1.5), ('3840x2160', 2)]]
    initial = fingerprints()
    def run(case):
        theme, size, scale = case
        key = f"{theme}-{size}-{format(scale, 'g').replace('.', 'p')}"
        output = base / key
        output.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()), '--theme', theme, '--size', size,
                   '--scale', str(scale), '--platform', args.platform, '--output', str(output),
                   '--cycles', str(args.cycles if case == ('light', '1600x900', 1) else 12)]
        if args.video and case in [('light', '1600x900', 1), ('light', '1600x900', 1.25)]:
            command.append('--video')
        with (output / 'run.log').open('w', encoding='utf-8') as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        result_path = output / 'result.json'
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        row = dict(case=key, exit_code=completed.returncode, passed=result.get('passed', False),
                   source_changed=result.get('source_changed_during_run', True),
                   source_matches_matrix=result.get('source_fingerprints') == initial,
                   screenshots=len(result.get('screenshots', [])), navigation_cycles=result.get('navigation_cycles', 0),
                   visible_categories=len(result.get('visible_categories', [])), result=str(result_path),
                   error=result.get('error'))
        print(json.dumps(row, ensure_ascii=False), flush=True)
        return row
    rows = []
    with ThreadPoolExecutor(max_workers=max(1, min(4, args.workers))) as pool:
        for future in as_completed([pool.submit(run, case) for case in cases]):
            rows.append(future.result())
    changed = fingerprints() != initial
    summary = dict(created_at=datetime.now(timezone.utc).isoformat(), source_fingerprints=initial,
                   validation_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   source_changed_during_matrix=changed, cases=sorted(rows, key=lambda row: row['case']),
                   passed=not changed and all(row['passed'] and not row['source_changed'] and row['source_matches_matrix'] for row in rows))
    (base / 'matrix-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    args = arguments()
    raise SystemExit(matrix(args) if args.matrix else main(args))
