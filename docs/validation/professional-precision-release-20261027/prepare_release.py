"""Transfer this committed release snapshot through the Parallels private link."""
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib,json,sys,threading,uuid
ROOT=Path.cwd()
sys.path[:0]=[str(ROOT/'scripts/validation')]
import fifth_round_remote as remote
remote.WINDOWS_ROOT=r'C:\tmp\fdm-cu-live-professional-20261027'
OUT=ROOT/'docs/validation/professional-precision-release-20261027'
BUILD=ROOT/'build/professional-precision-release-20261027'
DEST=r'C:\tmp\fdm-professional-precision-release-20261027'
sources={'source.zip':BUILD/'source.zip','build.cmd':OUT/'build.cmd','run-build.ps1':OUT/'run-build.ps1'}
route='/'+uuid.uuid4().hex
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        name=self.path.removeprefix(route+'/') if self.path.startswith(route+'/') else ''
        path=sources.get(name)
        if path is None:self.send_error(404);return
        self.send_response(200);self.send_header('Content-Length',str(path.stat().st_size));self.end_headers()
        with path.open('rb') as handle:
            while chunk:=handle.read(1024*1024):self.wfile.write(chunk)
    def log_message(self,*args):pass
server=ThreadingHTTPServer(('10.211.55.2',0),Handler)
thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
rows=[]
try:
    code="$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; $r='"+DEST+"'; if(Test-Path $r){throw 'Fresh release directory already exists'}; New-Item -ItemType Directory $r | Out-Null\n"
    for name,path in sources.items():
        with path.open('rb') as handle:digest=hashlib.file_digest(handle,'sha256').hexdigest()
        rows.append(dict(path=name,bytes=path.stat().st_size,sha256=digest))
        code+=f"Invoke-WebRequest -UseBasicParsing -Uri 'http://10.211.55.2:{server.server_port}{route}/{name}' -OutFile (Join-Path $r '{name}'); if((Get-FileHash (Join-Path $r '{name}')).Hash.ToLowerInvariant() -ne '{digest}'){{throw 'Transfer hash mismatch: {name}'}}\n"
    code+=r"""Expand-Archive (Join-Path $r 'source.zip') -DestinationPath $r
& 'C:\Users\lishuyang\Downloads\fiber-diameter-measurement\.venv\Scripts\python.exe' (Join-Path $r 'scripts\sync_cu_live_windows.py') --verify $r
if($LASTEXITCODE -ne 0){throw 'Snapshot verification failed'}
New-Item -ItemType Directory (Join-Path $r 'build\isolated-config'),(Join-Path $r 'build\temp') -Force | Out-Null
Remove-Item -LiteralPath (Join-Path $r 'source.zip')
$p=Start-Process powershell -ArgumentList @('-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',(Join-Path $r 'run-build.ps1')) -PassThru -WindowStyle Hidden
@{root=$r;build_pid=$p.Id;started=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json
"""
    result=remote.run_powershell('Windows 11',code,timeout=300)
    (OUT/'prepare.log').write_text(result.stdout+'\n'+result.stderr)
    print(result.stdout)
finally:
    server.shutdown();server.server_close();thread.join(timeout=2)
(OUT/'source-transfer.json').write_text(json.dumps(dict(files=rows,http_server_stopped=True),indent=2)+'\n')
