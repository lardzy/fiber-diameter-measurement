from pathlib import Path
import ctypes,json,psutil
from ctypes import wintypes as w
u=ctypes.WinDLL('user32',use_last_error=True);k=ctypes.WinDLL('kernel32',use_last_error=True)
u.GetForegroundWindow.restype=w.HWND
u.GetWindowThreadProcessId.argtypes=[w.HWND,ctypes.POINTER(w.DWORD)]
u.IsIconic.argtypes=[w.HWND];u.IsWindowVisible.argtypes=[w.HWND]
u.GetWindowRect.argtypes=[w.HWND,ctypes.POINTER(w.RECT)]
fg=u.GetForegroundWindow();fgpid=w.DWORD();u.GetWindowThreadProcessId(fg,ctypes.byref(fgpid))
ps={p.pid:p for p in psutil.process_iter(['pid','cmdline']) if any(x.endswith('sampler-diagnostic.py') for x in p.info['cmdline'] or [])}
windows=[]
@ctypes.WINFUNCTYPE(w.BOOL,w.HWND,w.LPARAM)
def visit(hwnd,_):
 pid=w.DWORD();u.GetWindowThreadProcessId(hwnd,ctypes.byref(pid))
 if pid.value in ps:
  r=w.RECT();u.GetWindowRect(hwnd,ctypes.byref(r));windows.append({'pid':pid.value,'hwnd':hwnd,'visible':bool(u.IsWindowVisible(hwnd)),'minimized':bool(u.IsIconic(hwnd)),'foreground':hwnd==fg,'rect':[r.left,r.top,r.right,r.bottom]})
 return True
u.EnumWindows(visit,0)
result={'foreground_pid':fgpid.value,'windows':windows,'processes':[{'pid':p.pid,'priority':p.nice(),'cmdline':p.info['cmdline']} for p in ps.values()]}
d=json.loads(Path('build/opt4k/sampler-diagnostic/0-on/result.json').read_text())
result['first_result']={k:d[k] for k in ['generated_fps','distinct_painted_fps','metrics','background_automatic_latency','resources']}
Path('build/opt4k/sampler-diagnostic/window-observe.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'foreground_pid':fgpid.value,'windows':windows,'processes':result['processes'],'fps':[d['generated_fps'],d['distinct_painted_fps']],'metrics':{k:{q:v[q] for q in ['p50_ms','p95_ms','max_ms']} for k,v in d['metrics'].items()}}))
