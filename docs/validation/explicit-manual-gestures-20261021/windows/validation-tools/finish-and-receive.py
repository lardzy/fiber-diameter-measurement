from pathlib import Path
import json,subprocess,sys,time
from datetime import datetime,timezone
root=Path.cwd();temp=root/'.tmp/acquisition-explicit-gestures-20261021';py=str(root/'.venv/bin/python');wrapper=str(temp/'delivery.py')
windows=r'C:\tmp\fdm-cu-live-explicit-gestures-20261021'
dist=root/'dist/internal/explicit-manual-gestures-20261021';dist.mkdir(parents=True,exist_ok=True)
timings=[]
def run(phase,command):
 started=time.monotonic();started_at=datetime.now(timezone.utc).isoformat();print('START',phase,started_at,flush=True)
 with (temp/(phase+'-launch.log')).open('w') as f:r=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
 timings.append(dict(phase=phase,started_at=started_at,finished_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=round(time.monotonic()-started,3),exit_code=r.returncode))
 (temp/'delivery-phase-timings.json').write_text(json.dumps(timings,indent=2))
 print('END',phase,r.returncode,timings[-1]['elapsed_seconds'],flush=True)
 if r.returncode:sys.exit(r.returncode)
def receive(phase,relative,local):
 run(phase,[py,wrapper,'receive',windows+'\\'+relative,str(local)])
run('registration-restore',['prlctl','exec','Windows 11','powershell','-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',windows+r'\scripts\validation\manage_acquisition_seventh_round_installation.ps1','-Action','Restore'])
run('collect-evidence',[py,wrapper,'command','--code-file',str(temp/'windows-collect-evidence.ps1'),'--timeout','300'])
receive('receive-identity',r'build\seventh-round\package\installer-identity.json',temp/'installer-identity.json')
identity=json.loads((temp/'installer-identity.json').read_text(encoding='utf-8'))
assert identity['version']=='0.4.10' and identity['passed'] is True
receive('receive-screenshot',r'build\seventh-round\explorer-start\installed-window-printwindow.png',temp/'installed-window-printwindow.png')
receive('receive-evidence',r'build\seventh-round\explicit-gestures-validation-evidence.zip',dist/'windows-validation-evidence.zip')
receive('receive-installer',r'dist\installer\fiber-diameter-measurement-setup-0.4.10-no-area-models.exe',dist/('FDM-0.4.10-explicit-gestures-'+identity['build_id'][:12]+'.exe'))
