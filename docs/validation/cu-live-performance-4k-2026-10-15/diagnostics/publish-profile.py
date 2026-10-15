from pathlib import Path
import time,json,sys,threading,ctypes
from ctypes import wintypes
from scripts import benchmark_cu_live_ui as bench
from fdm.live_experiment.controller import LiveExperimentController
from fdm.live_experiment.evidence import EvidenceStore
from fdm.live_experiment.frames import FrameBudget
from fdm.live_experiment.models import ExperimentSession
from fdm.ui.live_experiment_workspace import LiveExperimentWorkspace
from concurrent.futures import ThreadPoolExecutor
k=ctypes.WinDLL('kernel32',use_last_error=True)
k.GetCurrentThread.restype=wintypes.HANDLE
k.QueryThreadCycleTime.argtypes=[wintypes.HANDLE,ctypes.POINTER(ctypes.c_ulonglong)];k.QueryThreadCycleTime.restype=wintypes.BOOL
handle=k.GetCurrentThread();owner=threading.get_ident();active=None;rows=[]
def stamp():
 v=ctypes.c_ulonglong();assert k.QueryThreadCycleTime(handle,ctypes.byref(v));return time.perf_counter_ns(),v.value

def instrument(cls,name):
 old=getattr(cls,name)
 def call(self,*args,**kwargs):
  if active is None or threading.get_ident()!=owner:return old(self,*args,**kwargs)
  t,c=stamp()
  try:return old(self,*args,**kwargs)
  finally:
   e,d=stamp();active['phases'].append({'phase':cls.__name__+'.'+name,'wall_ms':(e-t)/1e6,'cpu_cycles':d-c})
 setattr(cls,name,call)
for cls,name in [(EvidenceStore,'reserve'),(FrameBudget,'acquire_with_scratch'),(ThreadPoolExecutor,'submit'),(ExperimentSession,'add_measurement'),(LiveExperimentController,'_ensure_field'),(LiveExperimentWorkspace,'_record_accepted'),(LiveExperimentWorkspace,'_changed')]:instrument(cls,name)
old=LiveExperimentController._accept
def accept(self,intent,result):
 global active
 row={'intent_id':intent.id,'phases':[]};active=row;t,c=stamp()
 try:return old(self,intent,result)
 finally:
  e,d=stamp();row.update(wall_ms=(e-t)/1e6,cpu_cycles=d-c);rows.append(row);active=None
LiveExperimentController._accept=accept
if __name__ == '__main__':
 out=Path('build/opt4k/refine');out.mkdir(exist_ok=True)
 mode=sys.argv[1]
 args=['--output',str(out/(mode+'.json')),'--sizes','2048','4096','--samples','30','--warmup','5','--evidence','on','--motion-input','lsb-noise' if mode=='dynamic-store' else 'stable']
 if mode=='joint-stable-profile':args+=['--background-automatic','multiscale']
 bench.main(args)
 (out/(mode+'-phases.json')).write_text(json.dumps({'rows':rows,'instrumentation':'Nested wall and QueryThreadCycleTime phases add diagnostic overhead; not formal acceptance.'},indent=2),encoding='utf8')
 for row in sorted(rows,key=lambda r:r['wall_ms'],reverse=True)[:8]:print(json.dumps(row),flush=True)
