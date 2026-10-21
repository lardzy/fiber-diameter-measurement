"""Record sixth-round manual input through real Qt navigation (synthetic camera).

The previous-failure case deliberately injects one controllable Future; normal
manual drag uses the real free-measurement pipeline. Video is annotated Qt
widget capture, not Windows desktop/native input or Microview acceptance.
"""
from __future__ import annotations

import argparse
from concurrent.futures import Future
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
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    os.environ['QT_SCALE_FACTOR'] = '1'
    sys.path.insert(0, str(ROOT / 'src'))
    from PySide6.QtCore import QEvent, QEventLoop, QPoint, QPointF, Qt, QTimer
    from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QDialogButtonBox, QInputDialog, QTableWidget
    from validate_workflow_redesign import WorkflowHarness, settle, synthetic_frame

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = dict(created_at=datetime.now(timezone.utc).isoformat(), runtime=platform.platform(),
                  qt_platform='offscreen', real_microview=False, real_windows=False, passed=False,
                  scope='Real application navigation and mouse events with isolated preferences and fixed synthetic camera frames.',
                  injection='Only delayed rejection uses a controlled Future; drag uses real free measurement.',
                  video_scope='Qt widget captures plus explanatory footer; no native desktop/input or performance claim.',
                  cases=[], source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                    for name in ('src/fdm/ui/live_measurement_canvas.py', 'src/fdm/ui/live_experiment_workspace.py',
                                 'src/fdm/live_experiment/controller.py', 'src/fdm/live_experiment/feedback.py')})
    encoder = None
    record_timer = None
    ticker = None
    caption = 'REAL Qt navigation | Synthetic source | Not a Windows or Microview recording'
    started = time.perf_counter()
    frames = 0
    try:
        with tempfile.TemporaryDirectory(prefix='fdm-round6-manual-') as temporary:
            with WorkflowHarness(Path(temporary), theme='light', size=(1366, 768)) as harness:
                window = harness.window
                width, height = window.width(), window.height() + 40
                encoder = subprocess.Popen([shutil.which('ffmpeg') or 'ffmpeg', '-y', '-loglevel', 'error',
                    '-f', 'rawvideo', '-pixel_format', 'rgba', '-video_size', f'{width}x{height}', '-framerate', '8',
                    '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '25', '-pix_fmt', 'yuv420p',
                    str(output / 'manual-input-qt-offscreen.mp4')], stdin=subprocess.PIPE, stderr=subprocess.PIPE)

                def image():
                    result = QImage(width, height, QImage.Format.Format_RGBA8888)
                    result.fill(QColor('#f4f4f4'))
                    painter = QPainter(result)
                    painter.drawPixmap(0, 0, window.grab())
                    painter.setPen(QColor('#142d47'))
                    painter.drawText(12, height - 15, caption)
                    painter.end()
                    return result

                def record():
                    nonlocal frames
                    frame = image()
                    encoder.stdin.write(bytes(frame.constBits()))
                    frames += 1

                def pause(milliseconds):
                    loop = QEventLoop()
                    QTimer.singleShot(milliseconds, loop.quit)
                    loop.exec()

                def until(predicate, label, seconds=10):
                    deadline = time.perf_counter() + seconds
                    while not predicate() and time.perf_counter() < deadline:
                        pause(25)
                    assert predicate(), (label, page.statusBar().currentMessage(), harness.slot_errors)

                def combo(combo, value):
                    index = combo.findData(value) if not isinstance(value, int) else value
                    combo.setFocus()
                    QTest.keyClick(combo, Qt.Key.Key_Home)
                    for _ in range(index):
                        QTest.keyClick(combo, Qt.Key.Key_Down)
                    QTest.keyClick(combo, Qt.Key.Key_Tab)
                    settle()
                    assert combo.currentIndex() == index

                record_timer = QTimer(window)
                record_timer.setInterval(125)
                record_timer.timeout.connect(record)
                record_timer.start()
                harness.enter()
                source = synthetic_frame(sequence=0)
                ticker = QTimer(window)
                ticker.setInterval(33)
                ticker.timeout.connect(lambda: harness.backend.tick(image=source))
                ticker.start()
                page = harness.select(1)

                def rename():
                    dialog = QApplication.activeModalWidget()
                    assert isinstance(dialog, QInputDialog)
                    dialog.setTextValue('手工事件验证')
                    dialog.accept()
                QTimer.singleShot(0, rename)
                page.rename_operator_button.trigger()

                def categories():
                    dialog = QApplication.activeModalWidget()
                    table = dialog.findChild(QTableWidget)
                    for row in range(2):
                        table.cellWidget(row, 0).setCurrentIndex(row + 1)
                        table.cellWidget(row, 3).setValue(0)
                        table.cellWidget(row, 4).setValue(0)
                    buttons = dialog.findChild(QDialogButtonBox)
                    QTest.mouseClick(buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.MouseButton.LeftButton)
                QTimer.singleShot(0, categories)
                QTest.mouseClick(page.choose_categories_button, Qt.MouseButton.LeftButton)
                combo(page.ruler_choice, 1)
                combo(page.measure_mode, 'free')
                assert page.canvas.input_mode == 'mixed'
                until(lambda: page.controller.state == 'ready' and page.analyze_button.isEnabled(), 'ready')

                def point(x, y):
                    scale, offset = page.canvas._transform()
                    return QPoint(round(x * scale + offset[0]), round(y * scale + offset[1]))

                def move(x, y, held=False):
                    position = point(x, y)
                    event = QMouseEvent(QEvent.Type.MouseMove, QPointF(position), QPointF(page.canvas.mapToGlobal(position)),
                        Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton if held else Qt.MouseButton.NoButton,
                        Qt.KeyboardModifier.NoModifier)
                    QApplication.sendEvent(page.canvas, event)

                def click(x, y):
                    QApplication.setActiveWindow(window)
                    QTest.mouseClick(page.canvas, Qt.MouseButton.LeftButton, pos=point(x, y))

                def snapshot(name, **details):
                    settle()
                    image().save(str(output / f'{name}.png'))
                    row = dict(name=name, elapsed_seconds=time.perf_counter() - started,
                               automatic=page.auto_analysis.isChecked(), draft=page.controller.draft is not None,
                               records=len(page.session.measurements), overlay_visible=page.measurement_feedback.isVisible(),
                               outcome=page.measurement_outcome.text(), status=page.statusBar().currentMessage(),
                               input_trace=page.controller.performance_frame_context()['input_trace'], **details)
                    report['cases'].append(row)

                for automatic in (False, True):
                    label = 'auto-on' if automatic else 'auto-off'
                    if page.auto_analysis.isChecked() != automatic:
                        QTest.mouseClick(page.auto_analysis, Qt.MouseButton.LeftButton)
                    assert page.controller.automatic_enabled == automatic
                    if automatic:
                        until(lambda: page.controller._automatic_context is not None and
                              page.controller._automatic_context.get('completed'), 'automatic analysis complete')
                    before = len(page.session.measurements)
                    caption = f'{label} | One click establishes first point; no second click is sent'
                    click(440, 390)
                    draft = page.controller.draft
                    assert draft is not None
                    pause(500)
                    snapshot(f'{label}-first-point')
                    for step in range(20):
                        move(440 + step * 3, 390 + step * 2)
                        pause(50)
                    for step in range(980):
                        move(500 + step % 40, 425)
                    caption = f'{label} | 1,000 NoButton moves: draft retained, no task submitted'
                    pause(700)
                    assert page.controller.draft is draft and not page.controller._jobs
                    assert len(page.session.measurements) == before
                    snapshot(f'{label}-1000-unpressed-moves', no_button_moves=1000)
                    QTest.keyClick(page.canvas, Qt.Key.Key_Escape)
                    assert page.controller.draft is None

                    caption = f'{label} | Real held-left drag: one manual measurement is submitted'
                    QTest.mousePress(page.canvas, Qt.MouseButton.LeftButton, pos=point(150, 260))
                    for step in range(1, 9):
                        move(150 + step * 5, 260 + step * 5, held=True)
                        pause(60)
                    QTest.mouseRelease(page.canvas, Qt.MouseButton.LeftButton, pos=point(190, 300))
                    until(lambda: len(page.session.measurements) == before + 1, 'drag accepted')
                    pause(500)
                    snapshot(f'{label}-held-drag-accepted')

                    caption = f'{label} | Controlled Future injection: previous task is waiting'
                    pending = Future()
                    with patch.object(page.controller._manual_pool, 'submit', lambda *_: pending):
                        click(350, 200)
                        click(390, 220)
                    intent = page.controller._jobs[-1][0]
                    click(520, 330)
                    draft = page.controller.draft
                    assert draft is not None
                    outcome = page.measurement_outcome.text()
                    pause(500)
                    caption = f'{label} | Injected old failure: new first point remains; no failure card'
                    pending.set_result({'status': 'rejected', 'reason': 'endpoints_not_background'})
                    until(lambda: not page.controller._jobs, 'old failure consumed')
                    assert page.controller.draft is draft
                    assert not page.measurement_feedback.isVisible()
                    assert page.measurement_outcome.text() == outcome
                    pause(900)
                    snapshot(f'{label}-old-failure-new-draft', injected_task_id=intent.id,
                             injected_reason='endpoints_not_background')
                    QTest.keyClick(page.canvas, Qt.Key.Key_Escape)
                assert not harness.slot_errors, harness.slot_errors
                report['slot_errors'] = harness.slot_errors
                report['passed'] = True
                ticker.stop()
                record_timer.stop()
                record()
    finally:
        if record_timer is not None:
            try:
                record_timer.stop()
            except RuntimeError:
                pass
        if ticker is not None:
            try:
                ticker.stop()
            except RuntimeError:
                pass
        if encoder is not None:
            encoder.stdin.close()
            error = encoder.stderr.read().decode('utf-8', errors='replace')
            code = encoder.wait(timeout=20)
            report['video'] = dict(frames=frames, playback_fps=8, encoder_exit_code=code, encoder_error=error)
        report['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        (output / 'manual-input.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    assert report['passed'] and report['video']['encoder_exit_code'] == 0
    print(json.dumps({'passed': report['passed'], 'cases': len(report['cases']), 'frames': frames,
                      'output': str(output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
