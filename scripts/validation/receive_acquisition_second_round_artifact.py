"""Retrieve one task-owned Windows artifact over the Parallels private link."""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from sync_cu_live_windows import powershell, ps_quote

parser = argparse.ArgumentParser()
parser.add_argument('remote')
parser.add_argument('local', type=Path)
args = parser.parse_args()
if not args.remote.lower().startswith('c:\\tmp\\fdm-cu-live-second-round-20261017\\'):
    parser.error('Only the isolated task checkout is allowed')
local = args.local.resolve()
local.relative_to(ROOT)
if local.exists():
    parser.error('Output exists; choose a new evidence path')
local.parent.mkdir(parents=True, exist_ok=True)
partial = local.with_suffix(local.suffix + '.partial')
route = '/' + uuid.uuid4().hex
received = {}

class Handler(BaseHTTPRequestHandler):
    def do_PUT(self):
        count = int(self.headers.get('Content-Length', '0'))
        if self.path != route or not 0 < count < 1024**3:
            self.send_error(400)
            return
        digest = hashlib.sha256()
        with partial.open('wb') as output:
            remaining = count
            while remaining:
                block = self.rfile.read(min(1024 * 1024, remaining))
                if not block:
                    raise IOError('Incomplete upload')
                output.write(block)
                digest.update(block)
                remaining -= len(block)
        received.update(bytes=count, sha256=digest.hexdigest())
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'OK')
    def log_message(self, *_):
        pass

server = ThreadingHTTPServer(('10.211.55.2', 0), Handler)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
try:
    url = f'http://10.211.55.2:{server.server_port}{route}'
    code = f"$ErrorActionPreference='Stop'; $file={ps_quote(args.remote)}; $hash=(Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash; Invoke-WebRequest -UseBasicParsing -Method Put -Uri '{url}' -InFile $file | Out-Null; [Console]::WriteLine($hash)"
    result = powershell('Windows 11', code, timeout=600)
    expected = result.stdout.strip().splitlines()[-1].lower()
    if not received or received['sha256'] != expected:
        raise RuntimeError('Transferred hash mismatch')
    partial.replace(local)
    print(json.dumps({'remote': args.remote, 'local': str(local), **received, 'verified': True}))
finally:
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)
    partial.unlink(missing_ok=True)
