from pathlib import Path
import subprocess,sys,time,json
out=Path('build/opt4k/hour');out.mkdir(exist_ok=True)
args=[sys.executable,'-X','utf8','-u','scripts/validation/soak_cu_workspace.py','--size','4096','--duration-seconds','3600','--minimum-accepted','10000','--output',str(out/'workspace-4k-hour.json')]
print('START WINDOWS 4K ONE HOUR',flush=True)
with (out/'workspace-4k-hour.log').open('w',encoding='utf8') as f:
 child=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8',errors='replace',bufsize=1)
 for line in child.stdout:
  f.write(line);f.flush()
  if line.startswith('{'):print(line.strip(),flush=True)
 child.wait()
print('HOUR EXIT',child.returncode,flush=True);raise SystemExit(child.returncode)
