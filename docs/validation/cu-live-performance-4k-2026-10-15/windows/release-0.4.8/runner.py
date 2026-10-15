from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,time

root=Path.cwd()
out=root/'build/opt4k/release-0.4.8'
out.mkdir(exist_ok=True)
os.environ['PYTHONUTF8']='1'
manifest=json.loads((out/'source.json').read_text(encoding='utf8'))
for row in manifest['files']:
    assert hashlib.sha256((root/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
from fdm.version import __version__
assert __version__=='0.4.8',__version__
records=[]
def run(label,args):
    began=time.time()
    print('START',label,flush=True)
    log=out/(label+'.log')
    with log.open('w',encoding='utf8') as f:
        p=subprocess.Popen([sys.executable,'-X','utf8','-u',*args],stdout=f,stderr=subprocess.STDOUT)
        last=began
        while p.poll() is None:
            time.sleep(5)
            if time.time()-last>=45:
                lines=log.read_text(encoding='utf8',errors='replace').splitlines()
                print('RUNNING',label,round(time.time()-began),lines[-1][:180] if lines else '',flush=True)
                last=time.time()
    row=dict(label=label,args=args,exit_code=p.returncode,seconds=time.time()-began)
    records.append(row)
    (out/'run.json').write_text(json.dumps(records,indent=2),encoding='utf8')
    print('DONE',label,p.returncode,round(row['seconds'],1),flush=True)
    if p.returncode:
        print(log.read_text(encoding='utf8',errors='replace')[-4000:],flush=True)
        raise SystemExit(p.returncode)

run('version-build-regression',['-m','pytest','-q','tests/test_build_support.py','tests/test_build_windows_onedir.py','tests/test_build_windows_installer.py','--junitxml='+str(out/'version-build-regression.xml')])
run('full-build',['scripts/build_windows_onedir.py','--profile','full','--exclude-area-models'])
shutil.copyfile(root/'build/self-check/packaged-runtime.json',out/'full-selfcheck.json')
selfcheck=json.loads((out/'full-selfcheck.json').read_text(encoding='utf8'))
assert selfcheck['ok'] and selfcheck['functional_checks']['cu_live_measurement']['revision']==4
assert all(selfcheck['functional_checks']['cu_live_measurement']['cases'].values())
shutil.copyfile(root/'dist/windows/FiberDiameterMeasurement/release-manifest.json',out/'full-release-manifest.json')
release=json.loads((out/'full-release-manifest.json').read_text(encoding='utf8'))
assert release['version']=='0.4.8' and release['profile']=='full'
assert release['excluded_components']==['area-models']
assert 'content-templates' in release['included_components']
run('full-code',['build/opt4k/release-0.4.8/verify-frozen-code.py','dist/windows/FiberDiameterMeasurement','--label','full-048'])
shutil.copyfile(root/'build/opt4k/full-048-frozen-code.json',out/'full-frozen-code.json')
installer=root/'dist/installer/fiber-diameter-measurement-setup-0.4.8-no-area-models.exe'
assert not installer.exists(),installer
run('installer',['scripts/build_windows_installer.py','--reuse-onedir','--exclude-area-models'])
identity=dict(path=str(installer),size=installer.stat().st_size,sha256=hashlib.file_digest(installer.open('rb'),'sha256').hexdigest(),version=__version__,profile='full',excluded_components=release['excluded_components'],included_components=release['included_components'])
(out/'installer-identity.json').write_text(json.dumps(identity,indent=2),encoding='utf8')
for row in manifest['files']:
    assert hashlib.sha256((root/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
print('RELEASE_VERIFIED',json.dumps(identity),flush=True)
