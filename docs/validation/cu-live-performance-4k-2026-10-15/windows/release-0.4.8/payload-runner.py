from pathlib import Path
import json,subprocess,sys,time,hashlib

root=Path.cwd();out=root/'build/opt4k/release-0.4.8'
destination=Path(r'C:\tmp\fdm-cu-live-unpacked-0.4.8-20261015')
extractor=Path(r'C:\tmp\fdm-cu-live-final-r4-20261015\build\validation-tools\innounp.exe')
installer=root/'dist/installer/fiber-diameter-measurement-setup-0.4.8-no-area-models.exe'
identity=json.loads((out/'installer-identity.json').read_text(encoding='utf8'))
assert installer.stat().st_size==identity['size']
assert hashlib.file_digest(installer.open('rb'),'sha256').hexdigest()==identity['sha256']
assert extractor.is_file(),extractor
assert hashlib.file_digest(extractor.open('rb'),'sha256').hexdigest()=='1c8453c198dd3fe7947f3705e59d2294ddce3dfa9f96a737e94449b119f44181'
assert not destination.exists(),destination
records=[]
def run(label,command,timeout=300):
    print('START',label,flush=True);began=time.time()
    log=out/(label+'.log')
    with log.open('w',encoding='utf8') as stream:
        p=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
        last=began
        while p.poll() is None:
            time.sleep(2)
            if time.time()-began>timeout:
                p.kill();p.wait();raise TimeoutError(label)
            if time.time()-last>=45:
                print('RUNNING',label,round(time.time()-began),flush=True);last=time.time()
    records.append(dict(label=label,args=command,exit_code=p.returncode,seconds=time.time()-began))
    (out/'payload-run.json').write_text(json.dumps(records,indent=2),encoding='utf8')
    print('DONE',label,p.returncode,round(time.time()-began,1),flush=True)
    if p.returncode:
        print(log.read_text(encoding='utf8',errors='replace')[-3000:],flush=True)
        raise SystemExit(p.returncode)

validator=str(out/'validate-extracted.py')
run('install-record-before',[sys.executable,validator,'before'])
run('extract-installer',[str(extractor),'-x','-b','-a','-u','-o','-d'+str(destination),str(installer)])
run('validate-extracted',[sys.executable,validator])
run('extracted-code',[sys.executable,str(out/'verify-frozen-code.py'),str(destination/'{app}'),'--label','extracted-048'])
run('extracted-gui',['powershell.exe','-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',str(out/'probe-gui.ps1'),'-AcknowledgeIsolatedEmptyProject'])
gui=list((out/'gui').glob('*/result.json'))
assert len(gui)==1 and json.loads(gui[0].read_text(encoding='utf8'))['ok']
print('INSTALLER_PAYLOAD_AND_GUI_VERIFIED',str(gui[0]),flush=True)
