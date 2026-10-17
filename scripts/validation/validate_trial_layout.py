"""Render the actual Qt trial page at supported logical sizes and DPI settings."""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage
from fdm.live_experiment.models import MaterialSnapshot


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    from fdm import settings, screenshot_settings, runtime_logging
    profile = args.output / "isolated-profile"
    profile.mkdir(exist_ok=True)
    settings.settings_file_path = lambda: profile / "settings.json"
    screenshot_settings.screenshot_settings_file_path = lambda: profile / "screenshot-settings.json"
    runtime_logging.runtime_log_path = lambda: profile / "startup.log"
    from fdm.ui.main_window import MainWindow
    from fdm.live_experiment.models import ExperimentSession
    app=QApplication.instance() or QApplication([])
    host=MainWindow()
    trial=ExperimentSession(); trial.add_operator('操作者 A')
    host.project.live_experiments.append(trial)
    host.show(); host.resize(1280,720)
    host._open_acquisition_workbench(trial_index=1)
    view=host._live_experiment_workspace
    names=['绵羊毛','山羊绒','棉','亚麻','苎麻','聚酯纤维','锦纶','粘胶纤维','其他纤维']
    materials=[MaterialSnapshot(str(i),name,1.3) for i,name in enumerate(names)]
    view.session.set_materials(materials)
    view._refresh_materials()
    image=QImage(768,576,QImage.Format.Format_RGB32); image.fill(0xffd6d2c4)
    from PySide6.QtGui import QPainter,QPen,QColor
    from PySide6.QtCore import QPointF
    painter=QPainter(image); painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    for i in range(8):
        painter.setPen(QPen(QColor(74+i*5,70+i*4,53+i*4),8+i%4))
        painter.drawLine(QPointF(i*100-100,0),QPointF(i*100+120,576))
    painter.end()
    from fdm.live_experiment.frames import frame_from_qimage
    frame=frame_from_qimage(image,source_id='layout-validation',sequence=1,live=False)
    view._receive_frame(frame)
    measurements=[]
    for width,height in ((1280,720),(1366,768)):
        host.resize(width,height); host.show(); host.activateWindow()
        for _ in range(5): app.processEvents(); time.sleep(.02)
        host.resize(width,height); app.processEvents()
        output=args.output/f'trial-{width}x{height}.png'
        host.grab().save(str(output))
        rows=[row['select'] for row in view.material_rows.values()]
        measurements.append({'requested':[width,height], 'actual':[host.width(),host.height()],
            'dpr':host.devicePixelRatioF(),'rows':len(rows),'last_row_bottom':rows[-1].mapTo(host,rows[-1].rect().bottomRight()).y(),
            'canvas':[view.canvas.width(),view.canvas.height()], 'screenshot':str(output),
            'physical_pixel_scale':view.canvas._transform()[0]*view.canvas.devicePixelRatioF(),
            'main_status_bar_visible':host.statusBar().isVisible(), 'canvas_tooltip':view.canvas.toolTip()})
        assert host.width()==width and host.height()==height and len(rows)==9
        assert measurements[-1]['last_row_bottom']<height-35
        assert measurements[-1]['physical_pixel_scale'] == 1
        assert not host.statusBar().isVisible() and not view.canvas.toolTip()
        from PySide6.QtTest import QTest
        for _ in range(20):
            host.acquisition_workspace_action.trigger()
            QTest.qWait(5)
            assert host._live_experiment_workspace is view
            assert host._acquisition_workbench.pages.currentWidget() is view
        from PySide6.QtCore import Qt
        assert not host._center_stack.testAttribute(Qt.WidgetAttribute.WA_NativeWindow)
        assert not host._acquisition_workbench.testAttribute(Qt.WidgetAttribute.WA_NativeWindow)
        screen_image = host.screen().grabWindow(int(host.winId()))
        if sys.platform == 'win32':
            assert not screen_image.isNull()
            screen_image.save(str(args.output/f'window-capture-{width}x{height}.png'))
        measurements[-1]['native_window_capture'] = not screen_image.isNull()
        view.advanced_toggle.setChecked(True)
        view.settings_dialog.resize(700, 520)
        for index in range(view.settings_tabs.count()):
            view.settings_tabs.setCurrentIndex(index)
            QTest.qWait(30)
            scroll = view.settings_tabs.currentWidget()
            assert not scroll.horizontalScrollBar().isVisible()
            assert scroll.widget().width() <= scroll.viewport().width()
            view.settings_dialog.grab().save(str(args.output/f'settings-{index}-{width}.png'))
        view.settings_dialog.close()
    host._confirm_close_documents_for_transition=lambda *_:True
    host.close(); app.processEvents()
    (args.output/'layout.json').write_text(json.dumps(measurements,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(measurements,ensure_ascii=False))

if __name__=='__main__': main()
