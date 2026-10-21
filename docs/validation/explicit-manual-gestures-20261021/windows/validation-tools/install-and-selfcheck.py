import subprocess,sys,json,time
from datetime import datetime,timezone
from pathlib import Path
root=Path.cwd();out=root/'.tmp/acquisition-explicit-gestures-20261021';py=str(root/'.venv/bin/python')
windows=r'C:\tmp\fdm-cu-live-explicit-gestures-20261021'
steps=[('registration-backup',['prlctl','exec','Windows 11','powershell','-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',windows+r'\scripts\validation\manage_acquisition_seventh_round_installation.ps1','-Action','Backup']),('actual-installation',['prlctl','exec','Windows 11','powershell','-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',windows+r'\scripts\validation\install_acquisition_seventh_round.ps1']),('ordinary-user-selfcheck',[py,'.tmp/acquisition-explicit-gestures-20261021/delivery.py','command','--code-file',str(out/'windows-installed-selfcheck.ps1'),'--timeout','1200'])]
timings=[]
for phase,command in steps:
 started=time.monotonic(); started_at=datetime.now(timezone.utc).isoformat()
 print('START',phase,started_at,flush=True)
 with (out/(phase+'-launch.log')).open('w') as log: result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
 timings.append(dict(phase=phase,started_at=started_at,finished_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=round(time.monotonic()-started,3),exit_code=result.returncode))
 (out/'installation-phase-timings.json').write_text(json.dumps(timings,indent=2))
 print('END',phase,result.returncode,timings[-1]['elapsed_seconds'],flush=True)
 if result.returncode:sys.exit(result.returncode)
