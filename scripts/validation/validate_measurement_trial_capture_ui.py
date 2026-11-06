"""Exercise measurement, capture and sample replacement in real Qt with synthetic pixels."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
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
    parser.add_argument('--scale', choices=(1., 1.25, 1.5, 2.), type=float, default=1.)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ['QT_QPA_PLATFORM'] = args.platform
    for key in ('QT_SCALE_FACTOR', 'QT_SCREEN_SCALE_FACTORS', 'QT_FONT_DPI'):
        os.environ.pop(key, None)
    baseline = float(subprocess.check_output([sys.executable, '-c',
        'from PySide6.QtWidgets import QApplication; a=QApplication([]); print(a.primaryScreen().devicePixelRatio())']))
    os.environ['QT_SCALE_FACTOR'] = str(args.scale / baseline)
    sys.path.insert(0, str(ROOT / 'src'))
    from PySide6.QtCore import QCoreApplication, QEvent, QPoint, QRect, Qt, QTimer
    from PySide6.QtGui import QColor, QImage, QPainter, QPen
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QMessageBox
    import numpy as np
    from fdm import runtime_logging, screenshot_settings, settings
    from fdm.geometry import Line, Point
    from fdm.measurement_annotation import MeasurementAnnotationLayout
    from fdm.models import Calibration, ImageDocument, Measurement
    from fdm.services.capture import CaptureBackend, CaptureDevice
    from fdm.settings import AppSettings, AppSettingsIO, MeasurementLabelStyleSettings
    from fdm.ui.main_window import MainWindow
    from fdm.ui.rendering import measurement_display_text_with_settings, resolve_measurement_annotation
    from fdm.ui.theme import semantic_colors
    from fdm.version import __version__

    app = QApplication([])
    result = dict(version=__version__, platform=args.platform, expected_dpr=args.scale,
                  actual_dpr=app.primaryScreen().devicePixelRatio(), passed=False, themes={}, screenshots={},
                  scope="Qt source runtime; synthetic preview backend, local profile and sample data")
    slot_errors = []
    sys.excepthook = lambda *exc: slot_errors.append(''.join(traceback.format_exception(*exc)))
    image = QImage(480, 300, QImage.Format.Format_RGB32)
    image.fill(QColor('#dfddd3'))
    painter = QPainter(image)
    for x in range(0, 500, 60):
        painter.setPen(QPen(QColor('#74736b'), 8))
        painter.drawLine(x, 270, x + 180, 30)
    painter.end()

    class Camera(CaptureBackend):
        backend_key = 'synthetic'
        starts = 0
        def list_devices(self):
            return [CaptureDevice('synthetic:source', '模拟相机', self.backend_key, 1)]
        def can_capture_still(self, device):
            return True
        def start_preview(self, device, *, frame_callback, **kwargs):
            self.starts += 1
            self.send = frame_callback
        def stop_preview(self):
            pass

    def settle():
        for _ in range(4):
            app.processEvents()

    def snap(widget, name):
        settle()
        path = output / (name + '.png')
        assert widget.grab().save(str(path)), name
        result['screenshots'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()

    def narrow(window, name, controls):
        window.resize(1067, 600)
        settle()
        assert window.size().toTuple() == (1067, 600), window.size()
        for control in controls:
            assert control.isVisible() and window.rect().contains(QRect(control.mapTo(window, QPoint()), control.size()))
        snap(window, name)
        window.resize(1360, 800)
        settle()

    def matching_pixels(widget, color, *, border=False):
        raster = widget.grab().toImage().convertToFormat(QImage.Format.Format_RGBA8888)
        rgb = np.frombuffer(raster.constBits(), dtype=np.uint8).reshape(raster.height(), raster.width(), 4)[:, :, :3]
        expected = QColor(color).getRgb()[:3]
        mask = (np.abs(rgb.astype(np.int16) - expected) <= 2).all(axis=2)
        if border:
            thickness = max(2, round(3 * raster.devicePixelRatioF()))
            mask[thickness:-thickness, thickness:-thickness] = False
        return int(np.count_nonzero(mask))

    try:
        assert abs(result['actual_dpr'] - args.scale) < 0.015
        for theme in ('light', 'dark'):
            profile = output / theme
            profile.mkdir()
            camera = Camera()
            with ExitStack() as stack:
                for mod, attr, filename in ((settings, 'settings_file_path', 'settings.json'),
                    (screenshot_settings, 'screenshot_settings_file_path', 'screenshot-settings.json'),
                    (runtime_logging, 'runtime_log_path', 'startup.log')):
                    stack.enter_context(patch.object(mod, attr, lambda name=filename: profile / name))
                stack.enter_context(patch('fdm.services.capture.available_capture_backends', lambda: [camera]))
                stack.enter_context(patch('fdm.services.motion_control.serial', None))
                stack.enter_context(patch('fdm.ui.main_window.list_motion_ports', lambda: []))
                AppSettingsIO.save(AppSettings(theme_mode=theme,
                    length_measurement_label_style=MeasurementLabelStyleSettings(parallel_to_line=True)))
                window = MainWindow()
                stack.enter_context(patch.object(window._screenshot_agent_client, 'status',
                    lambda **_: SimpleNamespace(running=False, error='', result={})))
                stack.enter_context(patch.object(window._screenshot_agent_client, 'send',
                    lambda *_, **__: SimpleNamespace(result={'running': False, 'offline': True, 'hotkeys': []})))
                window._confirm_close_documents_for_transition = lambda *_: True
                try:
                    window.resize(1360, 800)
                    window.show()
                    doc = ImageDocument("alignment-demo", str(profile/'样品示例.png'), (480, 300),
                                        calibration=Calibration(mode='preset', pixels_per_unit=10, unit='mm', source_label='模拟标尺'))
                    doc.initialize_runtime_state()
                    group = doc.create_group(color='#276891', label='样品甲')
                    other = doc.create_group(color='#af522c', label='样品乙')
                    line = Measurement("diameter-demo", doc.id, group.id, "manual",
                        line_px=Line(Point(92, 188), Point(152, 108)),
                        annotation_layout=MeasurementAnnotationLayout(placement_mode='manual',
                            offset_px=Point(59, 40), leader_style='dual'))
                    area = Measurement("area-demo", doc.id, other.id, "manual", "area",
                        polygon_px=[Point(295, 155), Point(329, 132), Point(348, 170), Point(308, 190)])
                    for record in (line, area):
                        record.recalculate(doc.calibration)
                        doc.add_measurement(record)
                    image.save(str(profile/'样品示例.png'))
                    window._mount_document(doc, image, tooltip=doc.path)
                    settle()
                    canvas = window.current_canvas()
                    canvas.fit_to_view()
                    caption = measurement_display_text_with_settings(line, doc, window._app_settings)
                    resolved = resolve_measurement_annotation(line, doc, window._app_settings, canvas.image_to_widget)
                    assert caption == '10.00 mm' and abs(resolved.angle) > 5, (caption, resolved.angle)
                    snap(window, theme + '-rotation')
                    snap(canvas, theme + '-rotation-canvas')
                    window._adaptive_layout.set_panel_visible("inspector", True)
                    window._inspector_dock.raise_()
                    window._show_inspector_section(window._object_properties_section)
                    window._records_controller.select_measurement_ids((line.id, area.id), primary=line.id)
                    settle()
                    editor = window._object_inspector.batch_editor
                    assert editor._group.currentIndex() == -1
                    snap(window, theme + '-batch')
                    editor._set_field('font_size', 18)
                    editor._apply.click()
                    assert doc.get_measurement(line.id).appearance.font_size == doc.get_measurement(area.id).appearance.font_size == 18
                    hub = window._open_acquisition_workbench(trial_index=0, auto_start=False)
                    settle()
                    colors = semantic_colors(app.palette())
                    emphasis_pixels = dict(start=matching_pixels(hub.capture_button, colors.accent))
                    snap(window, theme + '-ready')
                    hub.capture_button.click()
                    QTest.qWait(40)
                    camera.send(image)
                    settle()
                    controls = window._snapshot_controls
                    controls.grab_button.click()
                    assert controls.request is not None
                    camera.send(image)
                    QTest.qWait(80)
                    assert camera.starts == 1 and window._capture_manager.is_preview_active()
                    assert controls.count == 1 and controls.view.isVisible() and controls.notice.property('statusTone') == 'success'
                    emphasis_pixels.update(stop=matching_pixels(hub.capture_button, colors.accent, border=True),
                                           grab=matching_pixels(controls.grab_button, colors.accent))
                    snap(window, theme + '-snapshot')
                    narrow(window, theme + '-snapshot-narrow', (hub.capture_button, controls.grab_button, controls.view))
                    hub.select_page(1)
                    page = hub.pages.currentWidget()
                    page.sample.setText('合成样品 A')
                    page.sample_number.setText('A-01')
                    page.commit_pending_edits()
                    camera.send(image)
                    settle()
                    old = page.session.id
                    snap(window, theme + '-content')
                    emphasis_pixels['finish_border'] = matching_pixels(page.finish_trial_button, colors.warning, border=True)
                    narrow(window, theme + '-content-narrow', (hub.capture_button, page.finish_trial_button))
                    confirm_state = []
                    def confirm():
                        dialog = app.activeModalWidget()
                        if isinstance(dialog, QMessageBox):
                            confirm_state.append(dialog.defaultButton().text())
                            snap(dialog, theme + '-finish-confirm')
                            match = next((b for b in dialog.buttons() if b.text() == '移除并开始'), None)
                            if match is not None:
                                emphasis_pixels['remove'] = matching_pixels(match, colors.error)
                                emphasis_pixels['remove_area'] = match.grab().width() * match.grab().height()
                                emphasis_pixels['cancel_error_pixels'] = matching_pixels(dialog.defaultButton(), colors.error)
                                match.click()
                            else:
                                dialog.reject()
                    QTimer.singleShot(50, confirm)
                    page.finish_trial_button.click()
                    settle()
                    current = hub.pages.currentWidget()
                    assert confirm_state == ['取消'] and current.session.id != old, (confirm_state, old, current.session.id, current.statusBar().currentMessage(), page._closed)
                    assert not current.session.measurements and not current.session.count_adjustments and not current.sample.text()
                    assert camera.starts == 1 and window._capture_manager.is_preview_active()
                    camera.send(image)
                    settle()
                    snap(window, theme + '-blank-sample')
                    result['themes'][theme] = dict(caption=caption, rotation=resolved.angle,
                        capture_count=controls.count, camera_starts=camera.starts, blank_session=True,
                        emphasis_pixels=emphasis_pixels)
                    assert all(emphasis_pixels[k] >= 10 for k in ('start', 'stop', 'grab', 'finish_border')), emphasis_pixels
                    assert emphasis_pixels['remove'] > emphasis_pixels['remove_area'] * .25, emphasis_pixels
                    assert emphasis_pixels['cancel_error_pixels'] < 10, emphasis_pixels
                    assert not slot_errors, slot_errors
                finally:
                    window._reset_workspace()
                    window.close()
                    window.deleteLater()
                    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
                    settle()
        assert not slot_errors, slot_errors
        result['passed'] = True
    except Exception:
        result['traceback'] = traceback.format_exc()
        raise
    finally:
        result['slot_errors'] = slot_errors
        (output / 'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(passed=True, screenshots=len(result['screenshots']))))


if __name__ == '__main__':
    main()
