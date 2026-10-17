"""Stage and run the bounded Explorer launch probe over the Parallels private link."""
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import threading
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from sync_cu_live_windows import powershell, ps_quote


def main():
    data = (ROOT / "scripts/validation/probe_second_round_explorer.ps1").read_bytes()
    route = "/" + uuid.uuid4().hex
    remote = r"C:\tmp\fdm-cu-live-second-round-20261017\scripts\validation\probe_second_round_explorer.ps1"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != route:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("10.211.55.2", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://10.211.55.2:{server.server_port}{route}"
        download = f"$ErrorActionPreference='Stop'; Invoke-WebRequest -UseBasicParsing '{url}' -OutFile {ps_quote(remote)}; if((Get-FileHash {ps_quote(remote)}).Hash.ToLowerInvariant() -ne '{sha256(data).hexdigest()}'){{throw 'Probe transfer hash mismatch'}}"
        powershell("Windows 11", download, timeout=60)
        completed = powershell("Windows 11", "& " + ps_quote(remote), timeout=100)
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


if __name__ == "__main__":
    main()
