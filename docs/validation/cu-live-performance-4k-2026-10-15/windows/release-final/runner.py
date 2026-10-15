from pathlib import Path
import subprocess,sys,os,json,hashlib,shutil,time
root=Path.cwd();out=root/'build/opt4k/release-final';out.mkdir(exist_ok=True);os.environ['PYTHONUTF8']='1'
manifest=json.loads((root/'build/opt4k/optimized-source.json').read_text(encoding='utf8'))
for row in manifest['files']:assert hashlib.sha256((root/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
records=[]
def run(label,args):
 began=time.time();print('START',label,flush=True);log=out/(label+'.log')
 with log.open('w',encoding='utf8') as f:
  p=subprocess.Popen([sys.executable,'-X','utf8','-u',*args],stdout=f,stderr=subprocess.STDOUT)
  while p.poll() is None:
   time.sleep(10)
   if time.time()-began>50 and int(time.time()-began)%60<11:
    print('RUNNING',label,round(time.time()-began),log.read_text(encoding='utf8',errors='replace').splitlines()[-1:][0][:170] if log.stat().st_size else '',flush=True)
 row=dict(label=label,args=args,exit_code=p.returncode,seconds=time.time()-began);records.append(row)
 (out/'run.json').write_text(json.dumps(records,indent=2),encoding='utf8');print('DONE',label,p.returncode,round(row['seconds'],1),flush=True)
 if p.returncode:print(log.read_text(encoding='utf8',errors='replace')[-4000:]);raise SystemExit(p.returncode)
for profile in ('core','full'):
 run(profile+'-build',['scripts/build_windows_onedir.py','--profile',profile,*(['--exclude-area-models'] if profile=='full' else [])])
 shutil.copyfile(root/'build/self-check/packaged-runtime.json',out/(profile+'-selfcheck.json'))
 d=json.loads((out/(profile+'-selfcheck.json')).read_text(encoding='utf8'));assert d['ok'] and d['functional_checks']['cu_live_measurement']['revision']==4 and all(d['functional_checks']['cu_live_measurement']['cases'].values())
 shutil.copyfile(root/'dist/windows/FiberDiameterMeasurement/release-manifest.json',out/(profile+'-release-manifest.json'))
 run(profile+'-code',['build/opt4k/verify-frozen-code.py','dist/windows/FiberDiameterMeasurement','--label',profile])
 shutil.copyfile(root/'build/opt4k'/f'{profile}-frozen-code.json',out/f'{profile}-frozen-code.json')
 if profile=='core':
  target=out/'core-onedir';assert not target.exists();shutil.move(str(root/'dist/windows/FiberDiameterMeasurement'),str(target))
p=root/'dist/installer/fiber-diameter-measurement-setup-0.4.7-no-area-models.exe'
old=root/'build/opt4k/installer-attempt1.exe'
if p.exists() and not old.exists():p.rename(old)
run('installer',['scripts/build_windows_installer.py','--reuse-onedir','--exclude-area-models'])
id={'path':str(p),'size':p.stat().st_size,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()}
(out/'installer-identity.json').write_text(json.dumps(id,indent=2),encoding='utf8');print(id,flush=True)
for row in manifest['files']:assert hashlib.sha256((root/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
