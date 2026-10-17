import sys, time, threading, json, gc
from pathlib import Path
from tempfile import TemporaryDirectory
from PySide6.QtCore import Qt, QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication
from fdm.ui.project_save_coordinator import ProjectSaveCoordinator
from fdm.ui.project_save_self_check import _ProbeHost

app=QApplication.instance() or QApplication([])
log=[]
release=threading.Event()
entered=threading.Event()
main=threading.get_ident()
with TemporaryDirectory(prefix='fdm-save-lifetime-') as root:
 host=_ProbeHost();coordinator=ProjectSaveCoordinator(host)
 coordinator.request_save(str(Path(root)/'project.fdmproj'))
 def destroying(*_):
  log.append({'phase':'destroy_enter','thread':threading.get_ident()})
  entered.set()
  release.wait(3)
  log.append({'phase':'destroy_exit','thread':threading.get_ident()})
 coordinator._worker.destroyed.connect(destroying,Qt.ConnectionType.DirectConnection)
 end=time.monotonic()+2
 while not entered.is_set() and time.monotonic()<end:time.sleep(.001)
 # Run only pending GUI events; no blocking thread wait or deferred QObject deletion.
 start=time.perf_counter();app.processEvents();elapsed=(time.perf_counter()-start)*1000
 log.append({'phase':'gui_before_worker_destroyed','busy':coordinator.busy,'worker_retained':coordinator._worker is not None,'thread_retained':coordinator._thread is not None,'gui_ms':elapsed,'main':main})
 release.set()
 deadline=time.monotonic()+5
 while coordinator.busy and time.monotonic()<deadline:app.processEvents();time.sleep(.001)
 app.processEvents();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
 print(json.dumps(log),flush=True)
 coordinator.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
