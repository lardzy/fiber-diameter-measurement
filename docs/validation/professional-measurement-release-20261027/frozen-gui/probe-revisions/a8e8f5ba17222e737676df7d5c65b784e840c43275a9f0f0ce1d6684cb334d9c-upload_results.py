"""Run inside the owned Windows release directory to stream validated artifacts."""
from pathlib import Path
import hashlib,http.client,json,sys,zipfile
from urllib.parse import urlsplit
ROOT=Path(r'C:\tmp\fdm-professional-measurement-release-20261027')

def sha(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()

def send(base,name,path):
    parsed=urlsplit(base)
    if parsed.scheme!='http' or parsed.hostname!='10.211.55.2':raise ValueError('Only the Parallels host link is allowed')
    conn=http.client.HTTPConnection(parsed.hostname,parsed.port,timeout=180)
    conn.putrequest('PUT',parsed.path.rstrip('/')+'/'+name)
    conn.putheader('Content-Type','application/octet-stream');conn.putheader('Content-Length',str(path.stat().st_size));conn.endheaders()
    with path.open('rb') as handle:
        while chunk:=handle.read(1024*1024):conn.send(chunk)
    result=conn.getresponse();body=result.read();conn.close()
    if result.status!=200:raise RuntimeError((result.status,body))

identity=json.loads((ROOT/'build/package/installer-identity.json').read_text(encoding='utf-8'))
assert identity['passed']
gui=json.loads((ROOT/'build/gui-status.json').read_text(encoding='utf-8-sig'))
assert gui['exit_code']==0 and gui['independent_verification'] is True
probe=Path(gui['result']).resolve()
assert probe.is_relative_to((ROOT/'build/frozen-gui').resolve())
verified=json.loads((probe.parent/'verified.json').read_text(encoding='utf-8'))
assert verified['passed'] is True and verified['probe_sha256']==sha(probe)
installer=Path(identity['installer']);assert sha(installer)==identity['sha256']
rows=[];archive=ROOT/'build/validated-evidence.zip'
files=[]
for group in ['package','frozen-gui']:
    files.extend(p for p in (ROOT/'build'/group).rglob('*') if p.is_file() and not any(x in {'isolated-config','temp'} for x in p.relative_to(ROOT/'build').parts))
for name in ['release-build.log','release-status.json','verify.log','verify.stderr.log','verify-status.json','gui.log','gui.stderr.log','gui-status.json']:
    path=ROOT/'build'/name
    if path.is_file():files.append(path)
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as zipped:
    for path in sorted(set(files)):
        key=path.relative_to(ROOT/'build').as_posix();data=path.read_bytes();zipped.writestr(key,data)
        rows.append(dict(path=key,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
name=f"FDM-{identity['version']}-professional-measurement-{identity['build_id'][:12]}.exe"
meta=ROOT/'build/artifact-transfer.json'
meta.write_text(json.dumps(dict(installer=name,installer_sha256=sha(installer),installer_bytes=installer.stat().st_size,evidence_sha256=sha(archive),evidence_bytes=archive.stat().st_size,files=rows),indent=2)+'\n')
for token,path in [('installer.exe',installer),('evidence.zip',archive),('transfer.json',meta)]:send(sys.argv[1],token,path)
print(json.dumps(dict(installer=name,files=len(rows),sha256=identity['sha256'])))
