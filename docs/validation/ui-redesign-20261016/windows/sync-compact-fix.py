"""Synchronize only the compact-client fix and its validation into the owned VM."""
from pathlib import Path
import hashlib,json,sys,uuid,threading,zipfile,io
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'scripts'))
from sync_cu_live_windows import powershell
paths=['src/fdm/ui/main_window.py','src/fdm/ui/acquisition_workbench.py','tests/test_acquisition_compact_client.py','tests/test_acquisition_workflow_redesign.py','scripts/validation/validate_workflow_redesign.py','scripts/validation/probe_workflow_redesign_frozen.ps1']
if '--readme-only' in sys.argv:
 paths=['README.md']
rows=[{'path':p,'size':(ROOT/p).stat().st_size,'sha256':hashlib.sha256((ROOT/p).read_bytes()).hexdigest()} for p in paths]
apply='''from pathlib import Path
import hashlib,json,shutil
root=Path(r'C:\\tmp\\fdm-cu-live-ui-redesign-20261016')
manifest=root/'.fdm_snapshot.json'
backup=root/'.fdm_snapshot.before-small-height-fix.json'
assert not backup.exists()
shutil.copy2(manifest,backup)
s=json.loads(manifest.read_text(encoding='utf-8'))
changes=json.loads((root/'build/compact-overlay.json').read_text(encoding='utf-8'))
rows={r['path']:r for r in s['files']}
for r in changes:
 p=root/r['path'];assert p.stat().st_size==r['size'];assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'];rows[r['path']]=r
s['files']=sorted(rows.values(),key=lambda r:r['path'])
manifest.write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf-8')
count=0
for r in s['files']:
 if r['path'].startswith(('src/','native/','runtime/')):
  assert hashlib.sha256((root/r['path']).read_bytes()).hexdigest()==r['sha256'],r['path'];count+=1
print(json.dumps({'verified_production_files':count,'updated_files':changes,'source_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()}))
'''
if '--readme-only' in sys.argv:
 apply=apply.replace('.fdm_snapshot.before-small-height-fix.json','.fdm_snapshot.before-final-readme.json')
buffer=io.BytesIO()
with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
 for p in paths:z.write(ROOT/p,p)
 z.writestr('build/compact-overlay.json',json.dumps(rows))
 z.writestr('build/apply-compact-overlay.py',apply)
data=buffer.getvalue();route='/'+uuid.uuid4().hex
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  if self.path!=route:self.send_error(404);return
  self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
 def log_message(self,*_):pass
server=ThreadingHTTPServer(('10.211.55.2',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
try:
 remote=r'C:\tmp\fdm-cu-live-ui-redesign-20261016'
 code=f"$ErrorActionPreference='Stop'; $root='{remote}'; Invoke-WebRequest -UseBasicParsing 'http://10.211.55.2:{server.server_port}{route}' -OutFile \"$root\\build\\compact-overlay.zip\"; Expand-Archive -LiteralPath \"$root\\build\\compact-overlay.zip\" -DestinationPath $root -Force; & 'C:\\Users\\lishuyang\\Downloads\\fiber-diameter-measurement\\.venv\\Scripts\\python.exe' \"$root\\build\\apply-compact-overlay.py\"; exit $LASTEXITCODE"
 r=powershell('Windows 11',code);print(r.stdout);print(r.stderr)
finally:server.shutdown();server.server_close();thread.join(timeout=2)
