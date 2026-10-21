import json, os, sys, tempfile
from pathlib import Path
root=Path('/Users/lishuyang/PycharmProjects/fiber-diameter-measurement')
sys.path[:0]=[str(root/'src'),str(root/'scripts/validation')]
from PySide6.QtCore import Qt,QPoint
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication,QStyle,QStyleOptionComboBox
from validate_workflow_redesign import WorkflowHarness, settle
from fdm.live_experiment import built_in_materials
scale=float(os.environ.get('QT_SCALE_FACTOR','1'))
size=(round(1600/scale),round(900/scale)) if scale != 1 else (1366,768)
out=Path('/tmp/fdm-explicit-gesture-ui')/str(scale);out.mkdir(parents=True,exist_ok=True)
with tempfile.TemporaryDirectory(prefix='fdm-gesture-visual-') as profile:
 with WorkflowHarness(profile,theme='light' if scale==1 else 'dark',size=size) as h:
  h.enter();p=h.select(1)
  p.session.set_materials(built_in_materials('normal')[:9]);p.session.rename_operator('测试员');p._refresh_operators()
  p.ruler_choice.setCurrentIndex(1);p.ruler_choice.activated.emit(1);p._refresh_summary();settle()
  QTest.mouseClick(p.records_toggle,Qt.MouseButton.LeftButton);settle();h.assert_size()
  h.window.grab().save(str(out/'two-point.png'))
  initial=(p.image_scroll.geometry().getRect(),p.category_scroll.geometry().getRect(),h.hub.navigation.geometry().getRect())
  menu=p.manual_gesture_button.menu();menu.popup(p.manual_gesture_button.mapToGlobal(p.manual_gesture_button.rect().bottomLeft()));settle()
  menu.grab().save(str(out/'gesture-menu.png'))
  QTest.mouseClick(menu,Qt.MouseButton.LeftButton,pos=menu.actionGeometry(p.manual_gesture_actions['drag']).center());settle();h.assert_size()
  assert initial==(p.image_scroll.geometry().getRect(),p.category_scroll.geometry().getRect(),h.hub.navigation.geometry().getRect())
  h.window.grab().save(str(out/'drag.png'))
  opt=QStyleOptionComboBox();p.measure_mode.initStyleOption(opt)
  text_rect=p.measure_mode.style().subControlRect(QStyle.ComplexControl.CC_ComboBox,opt,QStyle.SubControl.SC_ComboBoxEditField,p.measure_mode)
  assert p.measure_mode.fontMetrics().horizontalAdvance(p.measure_mode.currentText()) <= text_rect.width()
  assert p.manual_gesture_button.text()=='拖线'
  assert not h.slot_errors and not h.modal_messages
  payload={'passed':True,'physical_scale':scale,'logical_size':size,'window':(h.window.width(),h.window.height()),'algorithm_text':p.measure_mode.currentText(),'algorithm_size':(p.measure_mode.width(),p.measure_mode.height()),'gesture_size':(p.manual_gesture_button.width(),p.manual_gesture_button.height()),'tool_group_size':(p.manual_tool_group.width(),p.manual_tool_group.height()),'category_size':(p.category_scroll.width(),p.category_scroll.height()),'header_size':(p.view_controls.width(),p.view_controls.height()),'scope':'macOS Qt offscreen; synthetic camera; real navigation and menu click; no hardware acceptance'}
  (out/'result.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
  print(json.dumps(payload,ensure_ascii=False))
