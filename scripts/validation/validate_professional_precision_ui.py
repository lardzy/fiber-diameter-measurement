"""Render real preference pages with isolated source-runtime Qt settings.

Each requested scale runs in a fresh process. Results describe widget rasters,
not physical monitor acceptance or an installed application.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--platform', default='windows' if sys.platform == 'win32' else 'offscreen')
    parser.add_argument('--scale', type=float, default=1)
    args = parser.parse_args()
    if args.scale <= 0:
        parser.error('scale must be positive')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ['QT_QPA_PLATFORM'] = args.platform
    env = os.environ.copy()
    for key in ('QT_SCALE_FACTOR', 'QT_SCREEN_SCALE_FACTORS', 'QT_FONT_DPI'):
        env.pop(key, None)
        os.environ.pop(key, None)
    baseline = float(subprocess.check_output([sys.executable, '-c',
        'from PySide6.QtWidgets import QApplication; a=QApplication([]); print(a.primaryScreen().devicePixelRatio())'],
        env=env, text=True).strip())
    os.environ['QT_SCALE_FACTOR'] = str(args.scale / baseline)
    sys.path.insert(0, str(ROOT / 'src'))
    from PySide6.QtCore import QPoint, QRect
    from PySide6.QtWidgets import QApplication
    from fdm.settings import AppSettings, MeasurementEndpointStyle
    from fdm.ui.dialogs import SettingsDialog
    from fdm.ui.theme import apply_application_theme
    from fdm.version import __version__
    app = QApplication([])
    report = dict(version=__version__, platform=args.platform, system=platform.platform(),
                  requested_scale=args.scale, baseline_dpr=baseline, passed=False, checks=[], screenshots=[],
                  scope='Real SettingsDialog source-runtime rendering; no user settings writes or hardware')
    def capture(dialog, name):
        app.processEvents()
        image = dialog.grab()
        path = output / (name + '.png')
        assert image.save(str(path))
        report['screenshots'].append(dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            pixels=[image.width(), image.height()], window=[dialog.width(), dialog.height()], dpr=image.devicePixelRatio()))
    try:
        for theme in ('light', 'dark'):
            apply_application_theme(app, theme)
            dialog = SettingsDialog(AppSettings(theme_mode=theme), document=None)
            try:
                dialog.select_page('professional'); dialog.show(); app.processEvents()
                for size in ((900, 640), (720, 520)):
                    dialog.resize(*size); app.processEvents()
                    scroll = dialog._settings_pages.currentWidget()
                    for decimals in (0, 2, 8):
                        dialog._professional_measurement_label_decimals.setValue(decimals)
                        app.processEvents()
                        assert dialog.app_settings().professional_measurement_label_decimals == decimals
                        preview = dialog._professional_measurement_style_preview
                        assert preview._decimals == decimals
                        for widget in (dialog._professional_measurement_label_decimals, preview):
                            assert scroll.viewport().rect().contains(QRect(widget.mapTo(scroll.viewport(), QPoint()), widget.size()))
                        capture(dialog, f'professional-{theme}-{size[0]}x{size[1]}-{decimals}')
                    report['checks'].append(dict(theme=theme, size=size, all_precisions_visible=True))
                dialog._restore_current_page_defaults()
                assert dialog.app_settings().professional_measurement_label_decimals == 2
                dialog.select_page('measurement'); app.processEvents()
                dialog._endpoint_style_combo.setCurrentIndex(dialog._endpoint_style_combo.findData(MeasurementEndpointStyle.ARROW_INSIDE))
                preview = dialog._length_measurement_style_preview
                scroll = dialog._settings_pages.currentWidget(); scroll.ensureWidgetVisible(preview, 0, 10); app.processEvents()
                for widget in (dialog._endpoint_style_combo, dialog._default_measurement_color, preview):
                    assert scroll.viewport().rect().contains(QRect(widget.mapTo(scroll.viewport(), QPoint()), widget.size()))
                assert preview._endpoint_style == MeasurementEndpointStyle.ARROW_INSIDE
                capture(dialog, f'measurement-merged-{theme}')
                dialog.select_page('screenshot')
                assert dialog._settings_page_title.text() == '截图工具'
                report['checks'].append(dict(theme=theme, restore_and_screenshot_key=True, legacy_preview_visible=True))
            finally:
                dialog.close(); dialog.deleteLater(); app.processEvents()
        report['passed'] = True
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['source_hashes'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), ROOT/'src/fdm/settings.py', ROOT/'src/fdm/ui/dialogs.py', ROOT/'src/fdm/ui/professional_rendering.py')}
        (output/'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'], screenshots=len(report['screenshots']), output=str(output))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
