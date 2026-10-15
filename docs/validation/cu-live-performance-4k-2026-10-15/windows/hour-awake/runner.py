from pathlib import Path
import ctypes,json,os,subprocess,sys,time
from datetime import datetime,timezone
out=Path('build/opt4k/hour-awake');out.mkdir(exist_ok=True)
stop=out/'stop.request'
assert not stop.exists()
k=ctypes.WinDLL('kernel32',use_last_error=True)
k.SetThreadExecutionState.argtypes=[ctypes.c_uint];k.SetThreadExecutionState.restype=ctypes.c_uint
previous=k.SetThreadExecutionState(0x80000003)
assert previous,ctypes.get_last_error()
metadata={'started_at_utc':datetime.now(timezone.utc).isoformat(),'validation_only_power_assertion':{'host':'caffeinate -di -t 4500, bound to runner lifetime','guest':'SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)','previous_state':previous,'settings_modified':False},'stop_file':str(stop.resolve())}
(out/'runner.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
try:
 args=[sys.executable,'-X','utf8','-u','scripts/validation/soak_cu_workspace.py','--size','4096','--duration-seconds','3600','--minimum-accepted','10000','--stop-file',str(stop),'--output',str(out/'workspace-4k-hour.json')]
 print('START AWAKE WINDOWS 4K ONE HOUR',flush=True)
 with (out/'workspace-4k-hour.log').open('w',encoding='utf8') as f:
  child=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8',errors='replace',bufsize=1)
  metadata.update(pid=child.pid,args=args);(out/'runner.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
  for line in child.stdout:
   f.write(line);f.flush()
   if line.startswith('{'):print(line.strip(),flush=True)
  child.wait()
 metadata.update(exit_code=child.returncode,finished_at_utc=datetime.now(timezone.utc).isoformat())
finally:
 metadata['guest_assertion_release_result']=k.SetThreadExecutionState(0x80000000)
 (out/'runner.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
print('HOUR EXIT',child.returncode,flush=True)
raise SystemExit(child.returncode)
