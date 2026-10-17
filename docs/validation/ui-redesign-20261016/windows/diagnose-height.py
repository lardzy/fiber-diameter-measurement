from pathlib import Path
import os,sys,json
os.environ['QT_QPA_PLATFORM']='windows'
os.environ['QT_SCALE_FACTOR']='0.5'
root=Path(r'C:\tmp\fdm-cu-live-ui-redesign-20261016')
sys.path[:0]=[str(root/'src'),str(root/'scripts/validation')]
from validate_workflow_redesign import WorkflowHarness,settle
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFontInfo

def pair(value):return [value.width(),value.height()]
def describe(w):
 layout=w.layout()
 return dict(type=type(w).__name__,name=w.objectName(),size=pair(w.size()),minimum=pair(w.minimumSize()),minimum_hint=pair(w.minimumSizeHint()),size_hint=pair(w.sizeHint()),has_hfw=w.hasHeightForWidth(),height_for_width=w.heightForWidth(w.width()),policy=[str(w.sizePolicy().horizontalPolicy()),str(w.sizePolicy().verticalPolicy())],layout_minimum=pair(layout.minimumSize()) if layout else None,font=QFontInfo(w.font()).family(),font_pixels=QFontInfo(w.font()).pixelSize())
with WorkflowHarness(root/'build/ui-redesign/source-height-profile',size=(1280,670)) as h:
 h.assert_size=lambda:None
 h.enter();h.select(1);settle(20)
 w=h.window;t=h.hub.pages.currentWidget()
 report={'font':QFontInfo(QApplication.font()).family(),'font_pixels':QFontInfo(QApplication.font()).pixelSize(),'dpr':w.devicePixelRatioF(),'items':{n:describe(o) for n,o in {'MainWindow':w,'center_stack':w._center_stack,'hub':h.hub,'pages':h.hub.pages,'trial':t,'body_splitter':t.body_splitter,'trial_sidebar':t.trial_sidebar,'workflow_tabs':t.workflow_tabs,'image_scroll':t.image_scroll,'canvas':t.canvas}.items()}}
 report['sidebar_children']=[describe(o) for o in t.trial_sidebar.findChildren(__import__('PySide6.QtWidgets',fromlist=['QWidget']).QWidget) if o.parentWidget() is t.trial_sidebar]
 (root/'build/ui-redesign/source-height-diagnosis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(report,ensure_ascii=True),flush=True)
