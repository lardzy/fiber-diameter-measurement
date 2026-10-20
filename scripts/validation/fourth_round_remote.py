"""Bounded Parallels script transport confined to the fourth-round checkout."""
from __future__ import annotations

import base64
import subprocess
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import uuid

import run_acquisition_second_round_frozen as shared

WINDOWS_ROOT = r"C:\tmp\fdm-cu-live-fourth-round-20261019"


def short_powershell(vm: str, code: str, *, timeout: int = 240):
    result = subprocess.run(
        ["prlctl", "exec", vm, "--current-user", "powershell", "-NoProfile",
         "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command",
         "$ProgressPreference='SilentlyContinue'; " + code],
        check=False, text=True, errors="replace", capture_output=True, timeout=timeout,
    )
    if result.returncode:
        raise RuntimeError(f"Windows command failed ({result.returncode}):\n{result.stdout}\n{result.stderr}")
    return result


# The legacy helper remains unchanged on disk. Its bounded upload calls use
# direct short commands for this process, avoiding UTF-16 base64 expansion.
shared.powershell = short_powershell


def run_powershell(vm: str, code: str, *, timeout: int):
    if len(base64.b64encode(code.encode("utf-16-le"))) <= 2000:
        return shared.powershell(vm, code, timeout=timeout)
    script = WINDOWS_ROOT + r"\build\fourth-round\runner-command-" + uuid.uuid4().hex + ".ps1"
    payload = b"\xef\xbb\xbf" + code.encode("utf-8")
    route = "/" + uuid.uuid4().hex

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != route:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("10.211.55.2", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        digest = hashlib.sha256(payload).hexdigest()
        download = ("$ErrorActionPreference='Stop'; Invoke-WebRequest -UseBasicParsing -Uri "
                    + shared.ps_quote(f"http://10.211.55.2:{server.server_port}{route}")
                    + " -OutFile " + shared.ps_quote(script)
                    + "; if((Get-FileHash -LiteralPath " + shared.ps_quote(script)
                    + ").Hash.ToLowerInvariant() -ne '" + digest + "'){throw 'Script hash mismatch'}")
        short_powershell(vm, download, timeout=60)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    try:
        # Direct -File avoids Parallels' intermittent rejection of an encoded
        # nested script invocation. Bypass is scoped to this owned process.
        result = subprocess.run(
            ["prlctl", "exec", vm, "--current-user", "powershell", "-NoProfile",
             "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", script],
            check=False, text=True, errors="replace", capture_output=True, timeout=timeout,
        )
        if result.returncode:
            raise RuntimeError(f"Windows command failed ({result.returncode}):\n{result.stdout}\n{result.stderr}")
        return result
    finally:
        shared.powershell(vm, "Remove-Item -LiteralPath " + shared.ps_quote(script)
                          + " -Force -ErrorAction SilentlyContinue", timeout=60)
