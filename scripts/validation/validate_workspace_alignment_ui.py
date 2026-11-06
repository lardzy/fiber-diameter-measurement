"""Source-runtime workspace isolation and acquisition alignment visual checks.

Run once per requested DPI in a fresh process. All frames are synthetic and all
settings/log paths are isolated under --output. No capture backend, serial port,
SDK, installed application, or physical measurement accuracy is exercised.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--platform', default='windows' if sys.platform == 'win32' else 'offscreen')
    parser.add_argument('--scale', type=float, choices=(1., 1.25, 1.5, 2.), default=1.)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ['QT_QPA_PLATFORM'] = args.platform
    for key in ('QT_SCALE_FACTOR', 'QT_SCREEN_SCALE_FACTORS', 'QT_FONT_DPI'):
        os.environ.pop(key, None)
    baseline = float(subprocess.check_output([sys.executable, '-c',
        'from PySide6.QtWidgets import QApplication; a=QApplication([]); print(a.primaryScreen().devicePixelRatio())'],
        env=os.environ.copy(), text=True).strip())
    os.environ['QT_SCALE_FACTOR'] = str(args.scale / baseline)
    sys.path.insert(0, str(ROOT / 'src'))
    from PySide6.QtCore import QCoreApplication, QEvent, QPoint, QRect, Qt
    from PySide6.QtGui import QColor, QImage, QPainter, QPen
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QMessageBox
    from fdm import runtime_logging, screenshot_settings, settings
    from fdm.models import ImageDocument
    from fdm.live_experiment.controller import CalibrationSnapshot
    from fdm.live_experiment.frames import frame_from_qimage
    from fdm.settings import AppSettings, AppSettingsIO
    from fdm.ui.acquisition_alignment import ALIGNMENT_FACTORS, aligned_offset
    from fdm.ui.main_window import MainWindow
    from fdm.version import __version__

    app = QApplication([])
    report = dict(version=__version__, platform=args.platform, system=platform.platform(),
        requested_scale=args.scale, baseline_dpr=baseline, actual_dpr=app.primaryScreen().devicePixelRatio(),
        passed=False, checks=[], screenshots=[], slot_errors=[], modal_messages=[],
        scope='Real Qt source-runtime windows and synthetic image frames; no capture SDK, installed build, physical monitor, or real-device acceptance')

    def settle():
        for _ in range(8):
            app.processEvents()

    def capture(widget, name):
        settle()
        raster = widget.grab()
        path = output / (name + '.png')
        assert raster.save(str(path))
        report['screenshots'].append(dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            pixels=[raster.width(), raster.height()], logical=[widget.width(), widget.height()], dpr=raster.devicePixelRatio()))

    image = QImage(768, 576, QImage.Format.Format_RGB32)
    image.fill(QColor('#e6e3d4'))
    painter = QPainter(image)
    painter.setPen(QPen(QColor('#8e8e85'), 1))
    for x in range(0, 768, 32):
        painter.drawLine(x, 0, x, 575)
    for y in range(0, 576, 32):
        painter.drawLine(0, y, 767, y)
    for rect, color in ((QRect(0, 0, 12, 576), '#c64747'), (QRect(756, 0, 12, 576), '#3556b8'),
                        (QRect(0, 0, 768, 12), '#3e8c48'), (QRect(0, 564, 768, 12), '#c3a527')):
        painter.fillRect(rect, QColor(color))
    painter.setPen(QPen(QColor('#333333'), 4))
    painter.drawEllipse(QPoint(384, 288), 110, 110)
    painter.drawLine(384, 100, 384, 476)
    painter.drawLine(196, 288, 572, 288)
    painter.end()
    image.save(str(output / 'synthetic-source.png'))
    source_sha = hashlib.sha256(bytes(image.constBits())).hexdigest()
    calibration = CalibrationSnapshot(.5, .5)

    def assert_panels_hidden(window):
        docks = (window._project_dock, window._inspector_dock, window._results_dock,
                 window.professional_workspace.dock)
        for dock in docks:
            assert dock.isHidden(), dock.objectName()
            assert not dock.toggleViewAction().isEnabled()
            assert not dock.toggleViewAction().isVisible()
            dock.show()
            dock.toggleViewAction().trigger()
        window.professional_workspace.show()
        popup = window.createPopupMenu()
        try:
            assert all(dock.toggleViewAction() not in popup.actions() for dock in docks)
        finally:
            popup.deleteLater()
        settle()
        assert all(dock.isHidden() for dock in docks)
        for action in (window.toggle_project_panel_action, window.toggle_inspector_panel_action,
                       window.toggle_results_panel_action, window._mode_actions['professional']):
            assert not action.isVisible()

    try:
        assert abs(report['actual_dpr'] - args.scale) < .01
        for theme in ('light', 'dark'):
            profile = output / ('profile-' + theme)
            profile.mkdir()
            with ExitStack() as stack:
                for module, key, filename in ((settings, 'settings_file_path', 'settings.json'),
                        (screenshot_settings, 'screenshot_settings_file_path', 'screenshot-settings.json'),
                        (runtime_logging, 'runtime_log_path', 'startup.log')):
                    stack.enter_context(patch.object(module, key, lambda name=filename: profile / name))
                stack.enter_context(patch('fdm.services.capture.available_capture_backends', lambda: []))
                stack.enter_context(patch('fdm.services.motion_control.serial', None))
                stack.enter_context(patch('fdm.ui.main_window.list_motion_ports', lambda: []))
                stack.enter_context(patch('sys.excepthook', lambda *exc:
                    report['slot_errors'].append(''.join(traceback.format_exception(*exc)))))
                def modal(_parent, title, text, *_args, **_kwargs):
                    report['modal_messages'].append(dict(title=title, text=text))
                    return QMessageBox.StandardButton.Cancel
                for name in ('warning', 'critical', 'information', 'question'):
                    stack.enter_context(patch.object(QMessageBox, name, modal))
                AppSettingsIO.save(AppSettings(theme_mode=theme))
                window = MainWindow()
                stack.enter_context(patch.object(window._screenshot_agent_client, 'status',
                    lambda **_: SimpleNamespace(running=False, error='', result={})))
                stack.enter_context(patch.object(window._screenshot_agent_client, 'send',
                    lambda *_args, **_kwargs: SimpleNamespace(result={'running': False, 'offline': True, 'hotkeys': []})))
                try:
                    window.show(); window.resize(1280, 720); settle()
                    window.professional_workspace.show(); settle()
                    assert not window.professional_workspace.dock.isHidden()
                    capture(window, f'{theme}-measurement-panels')
                    hub = window._open_acquisition_workbench(auto_start=False)
                    settle()
                    for index, name in enumerate(('capture', 'content', 'diameter')):
                        hub.select_page(index); settle()
                        assert_panels_hidden(window)
                        if index == 0:
                            canvas, binding, picker = window._preview_canvas, window._preview_display_binding, window._preview_alignment_button
                            preview = ImageDocument(id='validation-preview', path='synthetic-source.png', image_size=(768, 576))
                            window._preview_document = preview
                            with binding.mounting_frame():
                                canvas.set_document(preview, image)
                            binding._calibration_provider = lambda: calibration
                            binding.set_source('validation:synthetic-preview', (768, 576))
                            window._refresh_preview_surface()
                            page = None
                        else:
                            page = hub.pages.currentWidget()
                            frame = frame_from_qimage(image, source_id='validation:synthetic-trial', generation=1, sequence=index, live=False)
                            page._receive_frame(frame)
                            page.controller.set_calibration(calibration)
                            canvas, binding, picker = page.canvas, page.display_binding, page.alignment_button
                        binding._screen_pixel_mm_provider = lambda: .25
                        settle()
                        before_project = window.project.to_dict()
                        before_dirty = window._project_dirty()
                        before_session = page.session.to_dict() if page else None
                        before_calibration = page.controller.calibration if page else calibration
                        direction_checks = []
                        for mode, value in (('pixel', None), ('fit', None), ('percent', 175), ('magnification', 200)):
                            binding.set_mode(mode, value); settle()
                            for direction in ALIGNMENT_FACTORS:
                                picker.direction_buttons[direction].click(); settle()
                                assert picker.alignment() == direction
                                assert binding.store.preferences.alignment == direction
                                viewport = canvas if page is None else page.image_scroll.viewport()
                                if page is None:
                                    rect = canvas._base_image_target_rect()
                                    scale = canvas._zoom
                                    origin = (rect.x(), rect.y())
                                else:
                                    scale, offset = canvas._transform()
                                    local = canvas.mapTo(viewport, QPoint())
                                    origin = (local.x() + offset[0], local.y() + offset[1])
                                extent = (768 * scale, 576 * scale)
                                expected = aligned_offset(direction, (viewport.width(), viewport.height()), extent, canvas.devicePixelRatioF())
                                error = max(abs(actual - target) for actual, target in zip(origin, expected))
                                assert error <= 1 / canvas.devicePixelRatioF() + .01, (name, mode, direction, origin, expected)
                                if mode == 'pixel':
                                    assert abs(scale * canvas.devicePixelRatioF() - 1) < 1e-9
                                direction_checks.append(dict(mode=mode, direction=direction, scale=scale, position_error=error))
                        binding.set_mode('pixel'); settle()
                        capture(window, f'{theme}-{name}-1280x720')
                        picker.menu().popup(picker.mapToGlobal(QPoint(0, picker.height())))
                        settle()
                        menu = picker.menu()
                        for button in picker.direction_buttons.values():
                            assert menu.rect().contains(QRect(button.mapTo(menu, QPoint()), button.size()))
                        capture(menu, f'{theme}-{name}-nine-grid')
                        menu.close()
                        assert window.project.to_dict() == before_project
                        assert window._project_dirty() == before_dirty
                        assert hashlib.sha256(bytes(image.constBits())).hexdigest() == source_sha
                        if page:
                            assert page.session.to_dict() == before_session
                            assert page.controller.calibration == before_calibration
                        window.resize(1067, 600); settle()
                        assert (window.width(), window.height()) == (1067, 600)
                        if page:
                            page.records_toggle.click(); settle()
                            assert (window.width(), window.height()) == (1067, 600)
                            controls = page.view_controls
                            for control in (page.analyze_button, page.display_choice, picker):
                                assert controls.rect().contains(QRect(control.mapTo(controls, QPoint()), control.size()))
                        capture(window, f'{theme}-{name}-narrow')
                        if page and page.records_toggle.isChecked():
                            page.records_toggle.click(); settle()
                        window.resize(1280, 720); settle()
                        report['checks'].append(dict(theme=theme, page=name, panel_isolation=True, data_unchanged=True,
                            popup_unclipped=True, narrow_window_preserved=True, directions=direction_checks))
                    window.measure_workspace_action.trigger(); settle()
                    assert not window.professional_workspace.dock.isHidden()
                    capture(window, f'{theme}-measurement-restored')
                    assert not report['slot_errors'], report['slot_errors']
                    assert not report['modal_messages'], report['modal_messages']
                finally:
                    window._confirm_close_documents_for_transition = lambda *_: True
                    window.close(); window.deleteLater()
                    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
                    settle()
        assert not report['slot_errors'], report['slot_errors']
        assert not report['modal_messages'], report['modal_messages']
        report['passed'] = True
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        paths = [Path(__file__)] + [ROOT / 'src/fdm/ui' / name for name in (
            'main_window.py', 'workspace.py', 'professional_measurement.py', 'display_preferences.py',
            'acquisition_alignment.py', 'acquisition_alignment_button.py', 'acquisition_layout.py',
            'live_experiment_view.py', 'live_experiment_workspace.py', 'trial_assistance.py',
            'canvas.py', 'live_measurement_canvas.py')]
        report['source_hashes'] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        (output / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(passed=True, checks=len(report['checks']), screenshots=len(report['screenshots']), output=str(output))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
