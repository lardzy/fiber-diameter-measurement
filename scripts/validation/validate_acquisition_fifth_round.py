"""Fifth-round Qt navigation and candidate/evidence interaction acceptance.

The automatic analyzer, alignment, ledger and evidence writer are real. Camera
pixels come from a deterministic synthetic source, never from a Microview SDK.
Videos are Qt widget captures with app-owned windows composited, not desktop
recordings; this does not certify native window stacking or real device speed.
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
DEFAULT_OUTPUT = ROOT / '.tmp/acquisition-fifth-round-20261020/visual'


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--theme', choices=('light', 'dark', 'navy', 'warm'), default='light')
    parser.add_argument('--size', default='1600x900', help='Qt client raster, excluding taskbar and OS decorations')
    parser.add_argument('--scale', type=float, choices=(1, 1.25, 1.5, 2), default=1)
    parser.add_argument('--base-device-scale', type=float, default=1,
                        help='Native Qt DPR with QT_SCALE_FACTOR=1; e.g. 2 for a Windows display at 200%%')
    parser.add_argument('--platform', default='windows' if sys.platform == 'win32' else 'offscreen')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cycles', type=int, default=200)
    parser.add_argument('--menu-cycles', type=int, default=0)
    parser.add_argument('--motion-cycles', type=int, default=0)
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--matrix', action='store_true')
    parser.add_argument('--workers', type=int, default=2)
    return parser.parse_args()


def case_key(theme, size, scale):
    return f"{theme}-{size}-{format(scale, 'g').replace('.', 'p')}"


def fingerprints():
    names = ('src/fdm/ui/main_window.py', 'src/fdm/ui/live_experiment_workspace.py', 'src/fdm/ui/live_experiment_view.py',
             'src/fdm/ui/live_measurement_canvas.py', 'src/fdm/ui/live_measurement_annotations.py',
             'src/fdm/ui/trial_assistance.py', 'src/fdm/ui/measurement_evidence_view.py',
             'src/fdm/ui/measurement_evidence_gallery.py', 'src/fdm/live_experiment/performance.py',
             'src/fdm/live_experiment/controller.py', 'src/fdm/live_experiment/candidate_alignment.py',
             'src/fdm/live_experiment/evidence.py', 'src/fdm/live_experiment/evidence_options.py',
             'src/fdm/live_experiment/models.py')
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in names if (ROOT / name).exists()}


def main(args):
    if not 0 < args.base_device_scale <= 8:
        raise ValueError('--base-device-scale must be greater than zero and at most 8')
    os.environ['QT_QPA_PLATFORM'] = args.platform
    os.environ['QT_SCALE_FACTOR'] = str(args.scale / args.base_device_scale)
    sys.path.insert(0, str(ROOT / 'src'))
    from PySide6.QtCore import QEventLoop, QPoint, QPointF, Qt, QTimer
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QDialogButtonBox, QInputDialog, QPushButton, QTableWidget
    import numpy as np
    from validate_workflow_redesign import WorkflowHarness, dimensions, settle, synthetic_frame

    def pause(milliseconds):
        # QTest.qWait holds the Python GIL on some PySide builds and starves
        # multi-stage background alignment. Use the application event loop.
        loop = QEventLoop()
        QTimer.singleShot(milliseconds, loop.quit)
        loop.exec()

    physical = dimensions(args.size)
    logical = tuple(round(value / args.scale) for value in physical)
    output = (args.output or DEFAULT_OUTPUT / case_key(args.theme, args.size, args.scale)).resolve()
    output.mkdir(parents=True, exist_ok=True)
    profile = Path(tempfile.mkdtemp(prefix='profile-', dir=output))
    report = dict(schema_version=1, created_at=datetime.now(timezone.utc).isoformat(),
                  theme=args.theme, physical_client=physical, logical_client=logical, scale=args.scale,
                  base_device_scale=args.base_device_scale,
                  qt_scale_factor=args.scale / args.base_device_scale,
                  runtime=platform.platform(), platform=args.platform, real_hardware=False,
                  real_windows=sys.platform == 'win32', source_fingerprints=fingerprints(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  screenshots=[], interactions=[], passed=False,
                  limitations=['Real navigation, automatic worker, alignment, candidate commit and evidence writer.',
                               'Camera source is deterministic Qt-rendered synthetic fiber pixels.',
                               'Categories are filled through their actual modal editor; no candidate/result injection.',
                               'Qt scale factors do not test physical display migration or OS window stacking.',
                               'Video is Qt grab compositing, not a desktop or Microview capture.',
                               'This visual pass does not invoke Excel COM or certify capture throughput.'])
    h = None
    ticker = None
    recorder = dict(process=None, timer=None, frames=0, timeline=[], errors=[])
    started = time.perf_counter()
    try:
        with WorkflowHarness(profile, theme=args.theme, size=logical) as h:
            w = h.window
            h.enter()
            stable_image = synthetic_frame(sequence=0)
            ticker = QTimer(w)
            ticker.setInterval(35)
            ticker.timeout.connect(lambda: h.backend.tick(image=stable_image))
            ticker.start()
            report['font'] = h.font_report()
            report['dpr'] = w.devicePixelRatioF()
            raster = w.grab().size()
            report['actual_client_raster'] = [raster.width(), raster.height()]
            # A Windows VM may already have DPR=2 before QT_SCALE_FACTOR is
            # applied. Never label a 3200x1800 raster as the 1600x900 baseline.
            assert all(abs(actual - requested) <= 1 for actual, requested in
                       zip(report['actual_client_raster'], physical)), (
                'Qt client raster differs from requested physical size; set --base-device-scale '
                'to the native DPR', report['actual_client_raster'], physical, report['dpr'])

            def wait_until(predicate, description, timeout=12):
                deadline = time.perf_counter() + timeout
                while not predicate() and time.perf_counter() < deadline:
                    pause(20)
                    settle(2)
                if not predicate():
                    details = dict(slot_errors=h.slot_errors, modals=h.modal_messages)
                    if 'page' in locals():
                        details['status'] = page.statusBar().currentMessage()
                        controller = page.controller
                        alignment = controller._alignment.result
                        details['runtime'] = dict(candidates=len(controller.candidates), state=controller.state,
                            latest=controller.latest_frame.sequence if controller.latest_frame else None,
                            presented=controller.presented.frame.sequence if controller.presented else None,
                            aligned=alignment[1] if alignment else None,
                            alignment_valid=alignment[3].valid if alignment else None,
                            reference=controller._alignment.reference.frame_id if controller._alignment.reference else None,
                            source_context=controller._automatic_context['token'].frame.frame_id if controller._automatic_context else None,
                            performance=controller.performance_diagnostics(),
                            preview_pending=controller._candidate_preview_pending.sequence if controller._candidate_preview_pending else None,
                            preview_deadline=controller._candidate_preview_deadline, preview_interval=controller._preview_interval,
                            alignment_done=controller._alignment.future[0].done() if controller._alignment.future else None,
                            alignment_submitted=controller._alignment.future[3] if controller._alignment.future else None,
                            candidate_states=[candidate.get('state') for candidate in controller.candidates])
                    raise AssertionError(f'{description}: {details}')

            def rect(widget):
                origin = widget.mapTo(w, QPoint())
                return [origin.x(), origin.y(), widget.width(), widget.height()]

            navigation = rect(h.hub.navigation)
            def capture():
                result = w.grab()
                painter = QPainter(result)
                try:
                    for page in getattr(w, '_live_experiment_workspaces', {}).values():
                        dialog = page.evidence_review.dialog
                        if dialog.isVisible():
                            painter.drawPixmap(dialog.mapToGlobal(QPoint()) - w.mapToGlobal(QPoint()), dialog.grab())
                    gallery = getattr(w, '_evidence_gallery', None)
                    if gallery is not None and gallery.dialog.isVisible():
                        painter.drawPixmap(gallery.dialog.mapToGlobal(QPoint()) - w.mapToGlobal(QPoint()), gallery.dialog.grab())
                finally:
                    painter.end()
                return result

            if args.video:
                executable = shutil.which('ffmpeg')
                if executable is None:
                    raise RuntimeError('--video requires ffmpeg')
                raster = w.grab().size()
                recorder['process'] = subprocess.Popen([executable, '-y', '-loglevel', 'error', '-f', 'rawvideo',
                    '-pixel_format', 'rgba', '-video_size', f'{raster.width()}x{raster.height()}', '-framerate', '8',
                    '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '24', '-pix_fmt', 'yuv420p',
                    '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', str(output / 'candidate-evidence-navigation.mp4')],
                    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                recorder['timer'] = QTimer(w)
                recorder['timer'].setInterval(125)
                def video_frame():
                    try:
                        image = capture().toImage().convertToFormat(QImage.Format.Format_RGBA8888)
                        assert image.size() == raster
                        recorder['process'].stdin.write(image.constBits())
                        recorder['frames'] += 1
                    except Exception as exc:
                        recorder['errors'].append(str(exc))
                        recorder['timer'].stop()
                recorder['timer'].timeout.connect(video_frame)
                recorder['timer'].start()

            def snap(name, page=None):
                pause(140 if args.video else 60)
                settle()
                h.assert_size()
                assert rect(h.hub.navigation) == navigation and h.hub.navigation.isVisible()
                candidate_snapshot = (page is not None and page.canvas.candidate_visible and any(
                    item.get('state', 'available') in {'available', 'confirming'}
                    for item in page.controller.candidates))
                if candidate_snapshot:
                    # Capture the demonstrated candidate state after the actual
                    # displayed frame passes its exact-frame guard. Preview keeps
                    # streaming; no gate, result or frame is injected/frozen.
                    wait_until(page.canvas._candidates_visible, 'candidate overlay available for screenshot')
                    page.canvas.repaint()
                pixmap = capture()
                assert all(abs(actual - requested) <= 1 for actual, requested in
                           zip((pixmap.width(), pixmap.height()), physical))
                path = output / (name + '.png')
                assert pixmap.save(str(path))
                row = dict(name=name, path=str(path), raster=[pixmap.width(), pixmap.height()],
                           sha256=hashlib.sha256(path.read_bytes()).hexdigest(), navigation=rect(h.hub.navigation))
                if candidate_snapshot:
                    row['visible_candidate_labels'] = sum(key.startswith('candidate:') for key in page.canvas._label_placements)
                    assert row['visible_candidate_labels'] > 0, name
                # Check actual rendered pixels as well as widget geometry. A
                # visible canvas widget alone does not prove the frame drew.
                rgba = pixmap.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
                pixels = np.frombuffer(rgba.constBits(), np.uint8).reshape(rgba.height(), rgba.width(), 4)
                background = np.array([214, 210, 196], dtype=np.int16)
                row['visible_source_pixels'] = int(np.all(np.abs(pixels[:, :, :3].astype(np.int16) - background) <= 2, axis=2).sum())
                assert row['visible_source_pixels'] > 10000, (name, row['visible_source_pixels'])
                if page is not None:
                    widgets = (page.analyze_button, page.auto_analysis, page.candidate_display_button, page.display_choice)
                    for control in widgets:
                        assert control.isVisible(), control.objectName()
                        assert page.view_controls.rect().contains(control.mapTo(page.view_controls, QPoint()))
                        assert page.view_controls.rect().contains(control.mapTo(page.view_controls, control.rect().bottomRight()))
                    for index, control in enumerate(widgets):
                        bounds = control.geometry()
                        assert all(not bounds.intersects(other.geometry()) for other in widgets[index + 1:])
                    assert page.export_button.isVisible() and page.evidence_button.isVisible()
                    row.update(image=rect(page.image_scroll), controls={control.objectName() or control.accessibleName(): rect(control) for control in widgets},
                               visible_annotations=len(page.canvas._label_placements), candidate_count=len(page.controller.candidates))
                report['screenshots'].append(row)
                recorder['timeline'].append(dict(name=name, frame=recorder['frames'], elapsed=time.perf_counter() - started))

            def combo_index(combo, index):
                combo.setFocus()
                QTest.keyClick(combo, Qt.Key.Key_Home)
                for _ in range(index):
                    QTest.keyClick(combo, Qt.Key.Key_Down)
                QTest.keyClick(combo, Qt.Key.Key_Tab)
                settle()
                assert combo.currentIndex() == index

            def prepare_trial(page, categories):
                # Trigger the real operator dialog and fill its visible editor.
                def rename():
                    dialog = QApplication.activeModalWidget()
                    assert isinstance(dialog, QInputDialog)
                    dialog.setTextValue('Operator A')
                    dialog.accept()
                QTimer.singleShot(0, rename)
                page.rename_operator_button.trigger()
                if categories:
                    def edit_categories():
                        dialog = QApplication.activeModalWidget()
                        table = dialog.findChild(QTableWidget)
                        assert table is not None
                        for row in range(9):
                            table.cellWidget(row, 0).setCurrentIndex(row + 1)
                            # Deliberately unrestricted synthetic classes:
                            # this pass checks UI, not a biological range rule.
                            table.cellWidget(row, 3).setValue(0)
                            table.cellWidget(row, 4).setValue(0)
                        buttons = dialog.findChild(QDialogButtonBox)
                        QTest.mouseClick(buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.MouseButton.LeftButton)
                    QTimer.singleShot(0, edit_categories)
                    QTest.mouseClick(page.choose_categories_button, Qt.MouseButton.LeftButton)
                combo_index(page.ruler_choice, 1)
                wait_until(lambda: page.controller.state == 'ready' and page.analyze_button.isEnabled(), 'trial ready')

            h.select(0)
            snap('realtime')
            page = h.select(1)
            report['status_events'] = []
            page.controller.statusChanged.connect(lambda message: report['status_events'].append(str(message)))
            page.statusBar().messageChanged.connect(lambda message: report['status_events'].append('ui:' + str(message)))
            assert not page.material_rows and page.empty_categories.isVisible()
            assert not page.auto_analysis.isChecked()
            assert page.session.effective_evidence_options['mode'] == 'crop_jpeg'
            snap('content-empty', page)
            prepare_trial(page, True)
            vp = page.category_scroll.viewport()
            visible = [row['select'].text() for row in page.material_rows.values()
                       if vp.rect().contains(row['select'].mapTo(vp, QPoint()))
                       and vp.rect().contains(row['select'].mapTo(vp, row['select'].rect().bottomRight()))]
            report['visible_categories'] = visible
            required = 6 if args.size == '1600x900' and args.scale == 1 else 4 if args.size == '1600x900' and args.scale == 1.25 else 1
            assert len(visible) >= required, (logical, len(visible), required)
            snap('content-ready', page)
            QTest.mouseClick(page.analyze_button, Qt.MouseButton.LeftButton, pos=QPoint(12, page.analyze_button.height() // 2))
            report['after_analyze_click'] = dict(enabled=page.analyze_button.isEnabled(), controller_state=page.controller.state,
                worker=page.controller._automatic is not None, context=page.controller._automatic_context is not None,
                message=page.statusBar().currentMessage())
            wait_until(lambda: bool(page.controller.candidates) and page.controller.candidate_overlay_ready, 'real full-image candidates')
            assert page.workflow_tabs.currentIndex() == 0
            initial_ids = [item['candidate_id'] for item in page.controller.candidates]
            report['analysis'] = dict(candidate_count=len(initial_ids), algorithm=page.algorithm.currentData(), fixture=False)
            assert page.input_mode.currentData() == 'mixed'
            snap('content-candidates', page)
            if args.menu_cycles:
                # Open the real sidebar combo popup. Menus must reuse the
                # coherent presentation, not toggle overlays on each repaint.
                pause(600)
                visibility = []
                visibility_timer = QTimer(w)
                visibility_timer.setInterval(8)
                visibility_timer.timeout.connect(lambda: visibility.append(page.canvas._candidates_visible()))
                visibility_timer.start()
                batch = page.controller._automatic_context['batch_id']
                for _ in range(args.menu_cycles):
                    QTest.mouseClick(page.measure_mode, Qt.MouseButton.LeftButton,
                                     pos=QPoint(page.measure_mode.width() - 10, page.measure_mode.height() // 2))
                    pause(25)
                    QTest.keyClick(page.measure_mode, Qt.Key.Key_Escape)
                    pause(25)
                    assert page.controller._automatic_context['batch_id'] == batch
                visibility_timer.stop()
                report['menu_stress'] = dict(cycles=args.menu_cycles, samples=len(visibility),
                    visible_samples=sum(visibility), transitions=sum(a != b for a, b in zip(visibility, visibility[1:])))
                assert visibility and all(visibility), report['menu_stress']
            candidate = next(item for item in page.controller.candidates if item['state'] == 'available')
            center = tuple((a + b) / 2 for a, b in zip(candidate['start'], candidate['end']))
            def canvas_point():
                scale, offset = page.canvas._transform()
                return QPoint(round(center[0] * scale + offset[0]), round(center[1] * scale + offset[1]))
            # Scroll explicitly if actual-pixel mode places the candidate outside
            # a small/high-DPI viewport, preserving the user's scale choice.
            page.image_scroll.ensureVisible(canvas_point().x(), canvas_point().y(), 30, 30)
            settle()
            before = len(page.session.measurements)
            wait_until(lambda: page.controller.candidate_overlay_ready, 'candidate click gate')
            QTest.mouseClick(page.canvas, Qt.MouseButton.LeftButton, pos=canvas_point())
            wait_until(lambda: len(page.session.measurements) == before + 1, 'canvas candidate commit')
            assert [item['candidate_id'] for item in page.controller.candidates] == initial_ids
            assert page.workflow_tabs.currentIndex() == 0
            accepted = next(reversed(page.session.measurements.values()))
            wait_until(lambda: page.session.measurements[accepted.id].evidence['state'] == 'saved', 'local evidence written')
            report['evidence'] = page.session.measurements[accepted.id].evidence
            snap('content-partial-accepted', page)
            # Exercise the actual menu action and slider, never internal canvas state.
            page.candidate_display_menu.popup(page.candidate_display_button.mapToGlobal(QPoint(0, page.candidate_display_button.height())))
            settle()
            page.candidate_opacity.setFocus()
            QTest.keyClick(page.candidate_opacity, Qt.Key.Key_Home)
            assert page.candidate_opacity.value() == 20 and page.canvas.candidate_opacity == .2
            page.show_candidates_action.trigger()
            page.candidate_display_menu.close()
            assert not page.canvas.candidate_visible
            assert page.canvas.candidate_hits(QPointF(canvas_point())) == []
            snap('content-candidates-hidden', page)
            page.show_candidates_action.trigger()
            page.candidate_opacity.setValue(60)
            combo_index(page.measure_mode, page.measure_mode.findData('assisted'))
            # QTest delivers mouse events without asking the offscreen window
            # system to reactivate the owner after dismissing a popup.
            QApplication.setActiveWindow(w)
            QTest.mouseClick(page.canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier, pos=canvas_point())
            assert page.controller.draft is not None
            QTest.keyClick(page.canvas, Qt.Key.Key_Escape)
            assert page.controller.draft is None
            assert len(page.session.measurements) == before + 1
            assert page.input_mode.currentData() == 'mixed'
            QTest.mouseClick(page.latest_value, Qt.MouseButton.LeftButton)
            review = page.evidence_review
            wait_until(lambda: review.dialog.image.image_size is not None, 'evidence review display')
            QTest.mouseClick(review.dialog.pin, Qt.MouseButton.LeftButton)
            assert review.dialog.windowType() == Qt.WindowType.Tool
            assert not review.dialog.windowFlags() & Qt.WindowType.WindowStaysOnTopHint
            snap('evidence-pinned', page)
            review.dialog.grab().save(str(output / 'evidence-window.png'))
            QTest.mouseClick(review.dialog.annotations, Qt.MouseButton.LeftButton)
            assert all(not item.isVisible() for item in review.dialog.image.overlay_items)
            snap('evidence-clean-image', page)
            QTest.mouseClick(review.dialog.annotations, Qt.MouseButton.LeftButton)
            QTest.mouseClick(review.dialog.collapse, Qt.MouseButton.LeftButton)
            viewport_geometry = rect(page.image_scroll)
            QTest.mouseClick(page.evidence_button, Qt.MouseButton.LeftButton, pos=QPoint(8, page.evidence_button.height() // 2))
            gallery = w._evidence_gallery
            wait_until(lambda: gallery.decode_count > 0 and bool(gallery._cache), 'gallery lazy thumbnail')
            assert gallery.proxy.rowCount() == before + 1
            assert gallery.dialog.operator.currentData() is None
            assert gallery.dialog.direction.currentData() is None
            assert not gallery.dialog.isModal()
            assert not gallery.dialog.windowFlags() & Qt.WindowType.WindowStaysOnTopHint
            assert rect(page.image_scroll) == viewport_geometry
            gallery.dialog.grab().save(str(output / 'evidence-gallery.png'))
            snap('evidence-gallery', page)
            index = gallery.proxy.index(0, 0)
            target = gallery.dialog.view.visualRect(index).center()
            QTest.mouseClick(gallery.dialog.view.viewport(), Qt.MouseButton.LeftButton, pos=target)
            QTest.mouseDClick(gallery.dialog.view.viewport(), Qt.MouseButton.LeftButton, pos=target)
            wait_until(lambda: review.dialog.isVisible() and not review.preferences['follow'], 'gallery historical review')
            review.dialog.collapse.click()
            gallery.dialog.close()
            # Storage is editable from the actual advanced dialog and both
            # entrances immediately reflect the same future-record policy.
            menu = page.trial_tools_button.menu()
            QTimer.singleShot(30, lambda: QTest.mouseClick(menu, Qt.MouseButton.LeftButton,
                pos=menu.actionGeometry(page.advanced_toggle).center()))
            QTest.mouseClick(page.trial_tools_button, Qt.MouseButton.LeftButton)
            wait_until(lambda: page.settings_dialog.isVisible(), 'advanced evidence settings')
            initial_policy = page.session.effective_evidence_options['mode']
            combo_index(page.evidence_storage, page.evidence_storage.findData('full_png'))
            assert page.evidence_mode_actions['full_png'].isChecked()
            assert page.session.measurements[accepted.id].evidence['mode'] == initial_policy
            snap('advanced-evidence-options', page)
            page.settings_dialog.grab().save(str(output / 'advanced-evidence-options.png'))
            combo_index(page.evidence_storage, page.evidence_storage.findData(initial_policy))
            page.settings_dialog.close()
            # Discard via the prominent menu before a legitimate page switch;
            # pending-result confirmation is tested by the unit/business suite.
            action = next(action for action in page.analyze_button.menu().actions() if action.text().startswith('取消分析'))
            action.trigger()
            QTest.mouseClick(page.auto_analysis, Qt.MouseButton.LeftButton, pos=QPoint(8, page.auto_analysis.height() // 2))
            assert page.auto_analysis.isChecked()
            # The preceding manual analysis already consumed this stable
            # cycle. Start a new business field via its visible user action.
            new_field = next(button for button in page.findChildren(QPushButton) if button.text() == '新视场')
            QTest.mouseClick(new_field, Qt.MouseButton.LeftButton)
            wait_until(lambda: bool(page.controller.candidates) and page.controller.candidate_overlay_ready, 'automatic stable-field analysis')
            assert page.controller._automatic_context['origin'] == 'automatic'
            snapshot_batch = page.controller._automatic_context['batch_id']
            pause(400)
            assert page.controller._automatic_context['batch_id'] == snapshot_batch
            if args.motion_cycles:
                original_image = stable_image
                shifted = []
                for dx in (4, -6):
                    image = QImage(original_image.size(), original_image.format())
                    image.fill(Qt.GlobalColor.white)
                    painter = QPainter(image)
                    painter.drawImage(dx, 0, original_image)
                    painter.end()
                    shifted.append(image)
                batches = []
                for cycle in range(args.motion_cycles):
                    previous = page.controller._automatic_context['batch_id']
                    stable_image = shifted[0]
                    h.backend.tick(image=stable_image)
                    pause(45)
                    stable_image = shifted[1]
                    h.backend.tick(image=stable_image)
                    pause(45)
                    wait_until(lambda: page.controller._automatic_context is None or
                        page.controller._automatic_context['batch_id'] != previous, 'moving discards old batch')
                    stable_image = original_image
                    h.backend.tick(image=stable_image)
                    wait_until(lambda: bool(page.controller.candidates) and page.controller.candidate_overlay_ready,
                               'stable after consecutive movements')
                    current = page.controller._automatic_context['batch_id']
                    assert current != previous and current not in batches
                    batches.append(current)
                report['motion_stress'] = dict(cycles=args.motion_cycles, unique_batches=len(set(batches)))
            QTest.mouseClick(page.auto_analysis, Qt.MouseButton.LeftButton, pos=QPoint(8, page.auto_analysis.height() // 2))
            assert page.controller.candidates  # completed results remain usable
            snap('content-automatic-result', page)
            report['performance'] = page.controller.performance_diagnostics()
            action.trigger()
            diameter = h.select(2)
            prepare_trial(diameter, False)
            assert not diameter.auto_analysis.isChecked()
            snap('diameter-ready', diameter)
            for index in range(args.cycles):
                selected = index % 3
                h.select(selected)
                assert rect(h.hub.navigation) == navigation
                h.assert_navigation()
                if args.video and index < 9:
                    pause(130)
            report['navigation_cycles'] = args.cycles
            report['capture_starts'] = h.backend.starts
            assert h.backend.starts == 1
            page = h.select(1)
            assert not page.evidence_review.dialog.isVisible()
            assert page.canvas.candidate_input_enabled
            snap('content-return', page)
            assert not h.slot_errors, h.slot_errors
            assert not h.modal_messages, h.modal_messages
            allowed = {'MainWindow', 'QTipLabel', 'QMenu', 'QDialog', 'QInputDialog', '_EvidenceDialog', '_GalleryDialog', 'QComboBoxPrivateContainer'}
            unexpected = [event for event in h.observer.events if event['class'] not in allowed
                          and not (event.get('qt_class') == 'QComboBoxPrivateContainer'
                                   and event.get('window_type') == int(Qt.WindowType.Popup)
                                   and event.get('parent_class') == 'NoWheelComboBox')]
            assert not unexpected, unexpected
            report['window_events'] = h.observer.events
            report['interactions'] = ['category modal editor', 'real analyzer', 'candidate click commit', 'partial batch preserved',
                'opacity slider', 'hidden candidates noninteractive', 'manual draft and Escape',
                'pinned evidence and pure image', 'independent lazy gallery and locked history',
                'advanced storage policy sync', 'stable automatic analysis once', 'two-level navigation']
            report['passed'] = True
            ticker.stop()
            if recorder['timer'] is not None:
                recorder['timer'].stop()
    except Exception:
        report['error'] = traceback.format_exc()
        if h is not None:
            report['slot_errors'] = h.slot_errors
            report['modal_messages'] = h.modal_messages
    finally:
        if recorder['process'] is not None:
            process = recorder['process']
            process.stdin.close()
            code = process.wait(timeout=30)
            report['video'] = dict(path=str(output / 'candidate-evidence-navigation.mp4'), fps=8,
                frames=recorder['frames'], duration=recorder['frames'] / 8,
                kind='Qt widget grabs composited with application-owned evidence dialog', timeline=recorder['timeline'],
                errors=recorder['errors'], ffmpeg_exit=code, stderr=process.stderr.read().decode('utf-8', errors='replace'))
            if code or recorder['errors']:
                report['passed'] = False
        report['source_changed_during_run'] = fingerprints() != report['source_fingerprints']
        report['elapsed_seconds'] = round(time.perf_counter() - started, 3)
        (output / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'], result=str(output / 'result.json'), error=report.get('error')), ensure_ascii=False))
    return 0 if report['passed'] else 1


def matrix(args):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    base = (args.output or DEFAULT_OUTPUT).resolve()
    base.mkdir(parents=True, exist_ok=True)
    cases = [(theme, size, scale) for theme in ('light', 'dark', 'navy', 'warm')
             for size, scale in [('1600x900', 1), ('1600x900', 1.25), ('1600x900', 1.5), ('1366x768', 1)]]
    initial = fingerprints()
    def run(case):
        theme, size, scale = case
        key = case_key(*case)
        output = base / key
        output.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()), '--theme', theme, '--size', size,
                   '--scale', str(scale), '--platform', args.platform, '--output', str(output),
                   '--base-device-scale', str(args.base_device_scale),
                   '--cycles', str(args.cycles if case == ('light', '1600x900', 1) else 6),
                   '--menu-cycles', str(args.menu_cycles if case == ('light', '1600x900', 1) else 0),
                   '--motion-cycles', str(args.motion_cycles if case == ('light', '1600x900', 1) else 0)]
        if args.video and case in [('light', '1600x900', 1), ('light', '1600x900', 1.25)]:
            command.append('--video')
        with (output / 'run.log').open('w', encoding='utf-8') as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        result_file = output / 'result.json'
        result = json.loads(result_file.read_text(encoding='utf-8')) if result_file.exists() else {}
        row = dict(case=key, passed=result.get('passed', False), exit_code=completed.returncode,
                   source_changed=result.get('source_changed_during_run', True),
                   source_matches_matrix=result.get('source_fingerprints') == initial,
                   screenshots=len(result.get('screenshots', [])), navigation_cycles=result.get('navigation_cycles', 0),
                   visible_categories=len(result.get('visible_categories', [])), result=str(result_file), error=result.get('error'))
        print(json.dumps(row, ensure_ascii=False), flush=True)
        return row
    rows = []
    with ThreadPoolExecutor(max_workers=max(1, min(4, args.workers))) as pool:
        for future in as_completed([pool.submit(run, case) for case in cases]):
            rows.append(future.result())
    changed = fingerprints() != initial
    summary = dict(created_at=datetime.now(timezone.utc).isoformat(), source_fingerprints=initial,
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   source_changed_during_matrix=changed, cases=sorted(rows, key=lambda row: row['case']),
                   passed=not changed and all(row['passed'] and not row['source_changed'] and row['source_matches_matrix'] for row in rows))
    (base / 'matrix-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    args = arguments()
    raise SystemExit(matrix(args) if args.matrix else main(args))
