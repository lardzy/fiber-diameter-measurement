"""Receive only this release's verified installer and bounded evidence bundle."""
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import hashlib,json,sys,threading,uuid,zipfile
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'scripts/validation'))
import fifth_round_remote as remote
remote.WINDOWS_ROOT=r'C:\tmp\fdm-cu-live-professional-20261027'
OUT=ROOT/'docs/validation/professional-measurement-release-20261027'
STAGE=ROOT/'build/professional-measurement-release-20261027/transfer';STAGE.mkdir(exist_ok=True)
route='/'+uuid.uuid4().hex
limits={'installer.exe':600*1024*1024,'evidence.zip':96*1024*1024,'transfer.json':8*1024*1024}
class Handler(BaseHTTPRequestHandler):
    def do_PUT(self):
        name=self.path.removeprefix(route+'/') if self.path.startswith(route+'/') else ''
        size=int(self.headers.get('Content-Length','0'))
        if name not in limits or not 0<size<=limits[name]:self.send_error(400);return
        path=STAGE/(name+'.part');remaining=size
        with path.open('wb') as handle:
            while remaining:
                chunk=self.rfile.read(min(1024*1024,remaining))
                if not chunk:raise EOFError('Incomplete artifact transfer')
                handle.write(chunk);remaining-=len(chunk)
        path.replace(STAGE/name);self.send_response(200);self.end_headers();self.wfile.write(b'ok')
    def log_message(self,*args):pass
server=ThreadingHTTPServer(('10.211.55.2',0),Handler)
thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
try:
    code=f"$ErrorActionPreference='Stop'; $env:PYTHONUTF8='1'; & 'C:\\Users\\lishuyang\\Downloads\\fiber-diameter-measurement\\.venv\\Scripts\\python.exe' 'C:\\tmp\\fdm-professional-measurement-release-20261027\\build\\upload_results.py' 'http://10.211.55.2:{server.server_port}{route}'; if($LASTEXITCODE -ne 0){{throw 'Artifact upload failed'}}"
    result=remote.run_powershell('Windows 11',code,timeout=300)
    print(result.stdout);print(result.stderr)
finally:
    server.shutdown();server.server_close();thread.join(timeout=2)
def sha(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()
meta=json.loads((STAGE/'transfer.json').read_text())
assert sha(STAGE/'installer.exe')==meta['installer_sha256']
assert sha(STAGE/'evidence.zip')==meta['evidence_sha256']
assert Path(meta['installer']).name==meta['installer']
with zipfile.ZipFile(STAGE/'evidence.zip') as archive:
    expected={r['path']:r for r in meta['files']}
    assert set(archive.namelist())==set(expected)
    for name,row in expected.items():
        target=(OUT/name).resolve();assert target.is_relative_to(OUT.resolve())
        data=archive.read(name);assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
import shutil
retained=ROOT/'dist/internal/professional-measurement-20261027';retained.mkdir(exist_ok=True)
target=retained/meta['installer']
if target.exists():assert sha(target)==meta['installer_sha256']
else:shutil.copy2(STAGE/'installer.exe',target)
assert sha(target)==meta['installer_sha256']
(retained/'SHA256SUMS.txt').write_text(meta['installer_sha256']+'  '+target.name+'\n')
(OUT/'transfer.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
(OUT/'host-retained-installer.json').write_text(json.dumps(dict(path=str(target),sha256=sha(target),bytes=target.stat().st_size),ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(installer=str(target),sha256=sha(target),evidence_files=len(meta['files'])),ensure_ascii=False))
