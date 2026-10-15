#!/usr/bin/env python3
"""Copy a source snapshot to a separate Parallels validation checkout.

No operation touches the existing Windows checkout or its Git state.  The HTTP
server exposes one randomly named ZIP on the Parallels private interface only,
and is shut down as soon as the VM has received it.  The old checkout supplies
an already installed Python environment, never source files.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import uuid
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = r"C:\tmp\fdm-cu-live-20261015"
DEFAULT_PYTHON = r"C:\Users\lishuyang\Downloads\fiber-diameter-measurement\.venv\Scripts\python.exe"


def powershell(vm: str, code: str, *, timeout: int = 240) -> subprocess.CompletedProcess:
    code = "$ProgressPreference='SilentlyContinue';\n" + code
    encoded = base64.b64encode(code.encode("utf-16-le")).decode("ascii")
    result = subprocess.run(
        ["prlctl", "exec", vm, "--current-user", "powershell", "-EncodedCommand", encoded],
        check=False, text=True, errors="replace", capture_output=True, timeout=timeout,
    )
    if result.returncode:
        raise RuntimeError(f"Windows command failed ({result.returncode}):\n{result.stdout}\n{result.stderr}")
    return result


def ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def verify_snapshot(root: Path) -> None:
    manifest = json.loads((root / ".fdm_snapshot.json").read_text(encoding="utf-8"))
    old_path = root / ".fdm_snapshot.previous.json"
    if old_path.exists():
        old = json.loads(old_path.read_text(encoding="utf-8"))
        keep = {row["path"] for row in manifest["files"]}
        for item in old["files"]:
            relative = Path(item["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError("Unsafe previous manifest")
            if item["path"] not in keep and not item["path"].startswith("validation_reference/"):
                (root / relative).unlink(missing_ok=True)
        old_path.unlink()
    for item in manifest["files"]:
        data = (root / item["path"]).read_bytes()
        if len(data) != item["size"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise RuntimeError("Snapshot hash mismatch: " + item["path"])
    print(json.dumps({"verified_files": len(manifest["files"]), "destination": str(root), "head": manifest["head"]}, allow_nan=False))


def snapshot(path: Path, manual_reference: Path | None = None, automatic_reference: Path | None = None) -> dict:
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT,
    ).decode("utf-8").split("\0")
    files = []
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for name in sorted(set(names)):
            if not name or name.startswith(("runtime/", ".git/", "build/", "dist/")):
                continue
            source = ROOT / name
            if not source.is_file() or source.is_symlink():
                continue
            data = source.read_bytes()
            files.append({"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
            archive.writestr(name, data)
        if manual_reference is not None:
            audit_path = manual_reference / "docs/validation/diameter_manual_20261011.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            reference_files = {audit_path}
            for row in audit["differential_validation"]["records"]:
                raw = manual_reference / row["raw_run"]["path"].replace("\\", "/")
                reference_files.update((raw, raw.parent / "input.bmp"))
            for source in sorted(reference_files):
                # Only recorded JSON and BMP; never original executable code.
                if source.suffix.lower() not in {".json", ".bmp"}:
                    raise RuntimeError("Unexpected reference file type")
                name = "validation_reference/" + source.relative_to(manual_reference).as_posix()
                data = source.read_bytes()
                files.append({"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
                archive.writestr(name, data)
        if automatic_reference is not None:
            audit_path = automatic_reference / "docs/validation/diameter_demo_20261011.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            reference_files = {audit_path}
            for suite in audit["validation_suites"]:
                for row in suite["cases"]:
                    raw = automatic_reference / row["raw_run"]
                    reference_files.update((raw, raw.parent / "input.bmp"))
                    recorded = json.loads(raw.read_text(encoding="utf-8"))["results"]["original"]
                    for key, filename in recorded.get("images", {}).items():
                        if key in {"mask", "mask_labeled", "contour", "edges", "selection"}:
                            reference_files.add(raw.parent / "original" / filename)
            for source in sorted(reference_files):
                if source.suffix.lower() not in {".json", ".bmp"}:
                    raise RuntimeError("Unexpected automatic reference file type")
                name = "validation_reference/" + source.relative_to(automatic_reference).as_posix()
                data = source.read_bytes()
                files.append({"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
                archive.writestr(name, data)
        # pybind11's build helper and headers are platform-independent. Reuse
        # the host's installed copy in this validation directory, without pip
        # installs or changes to the user's existing Windows environment.
        spec = importlib.util.find_spec("pybind11")
        if spec is None or spec.origin is None:
            raise RuntimeError("The host validation environment needs pybind11")
        vendor_root = Path(spec.origin).parent
        for source in sorted(vendor_root.rglob("*")):
            if not source.is_file() or "__pycache__" in source.parts or source.suffix == ".pyc":
                continue
            name = "validation_vendor/pybind11/" + source.relative_to(vendor_root).as_posix()
            data = source.read_bytes()
            files.append({"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
            archive.writestr(name, data)
        manifest = {"format": 1, "source": str(ROOT), "files": files,
                    "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
        archive.writestr(".fdm_snapshot.json", json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vm", default="Windows 11")
    parser.add_argument("--destination", default=DEFAULT_DESTINATION)
    parser.add_argument("--bind", default="10.211.55.2")
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--manual-reference", type=Path, help="Optional recorded JSON/BMP CU evidence root")
    parser.add_argument("--automatic-reference", type=Path, help="Optional recorded automatic JSON/BMP evidence root")
    parser.add_argument("--verify", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.verify:
        verify_snapshot(args.verify)
        return
    if not re.fullmatch(r"C:\\tmp\\fdm-cu-live-[A-Za-z0-9_-]+", args.destination):
        parser.error("destination must be a separate C:\\tmp\\fdm-cu-live-* directory")
    interfaces = subprocess.check_output(["ifconfig"], text=True)
    if f"inet {args.bind} " not in interfaces or not args.bind.startswith("10.211.55."):
        parser.error("bind address must be an active Parallels private IPv4 interface")

    with tempfile.TemporaryDirectory(prefix="fdm-cu-sync-") as temporary:
        archive_path = Path(temporary) / "source.zip"
        manifest = snapshot(archive_path, args.manual_reference, args.automatic_reference)
        archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        route = "/" + uuid.uuid4().hex + ".zip"

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path != route:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Length", str(archive_path.stat().st_size))
                self.end_headers()
                with archive_path.open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        self.wfile.write(chunk)

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer((args.bind, 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://{args.bind}:{server.server_port}{route}"
        destination = ps_quote(args.destination)
        # Extract into a fixed independent directory, then verify every source
        # hash. Previously managed files absent in the new snapshot are removed;
        # build products and validation reports remain available between runs.
        code = f"""$ErrorActionPreference='Stop'
$destination={destination}
New-Item -ItemType Directory -Force -Path $destination | Out-Null
$archive=Join-Path $env:TEMP 'fdm-cu-{uuid.uuid4().hex}.zip'
try {{
 Invoke-WebRequest -UseBasicParsing -Uri {ps_quote(url)} -OutFile $archive
 if ((Get-FileHash -Algorithm SHA256 $archive).Hash.ToLowerInvariant() -ne '{archive_sha}') {{ throw 'Snapshot ZIP hash mismatch' }}
 $old=Join-Path $destination '.fdm_snapshot.json'
 if (Test-Path $old) {{ Copy-Item $old (Join-Path $destination '.fdm_snapshot.previous.json') -Force }}
 Expand-Archive -LiteralPath $archive -DestinationPath $destination -Force
 & {ps_quote(args.python)} (Join-Path $destination 'scripts\\sync_cu_live_windows.py') --verify $destination
 if ($LASTEXITCODE -ne 0) {{ throw 'Source snapshot verification failed' }}
}} finally {{ Remove-Item $archive -Force -ErrorAction SilentlyContinue }}
"""
        try:
            result = powershell(args.vm, code)
            print(result.stdout.strip(), flush=True)
            if result.stderr:
                print(result.stderr.strip(), flush=True)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
    print(json.dumps({"snapshot_files": len(manifest["files"]), "zip_sha256": archive_sha,
                      "http_server_stopped": True}, allow_nan=False), flush=True)
    if args.build:
        code = f"""$ErrorActionPreference='Stop'
$destination={destination}
$logs=Join-Path $destination 'build\\cu-live-validation'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$batch=Join-Path $logs 'build_native.cmd'
$lines=@('@echo off','call "C:\\BuildTools\\VC\\Auxiliary\\Build\\vcvars64.bat"','if errorlevel 1 exit /b %errorlevel%','cd /d "{args.destination}"','set PYTHONPATH={args.destination}\\src;{args.destination}\\validation_vendor','"{args.python}" setup.py build_ext --inplace > "{args.destination}\\build\\cu-live-validation\\native-build.log" 2>&1','exit /b %errorlevel%')
[IO.File]::WriteAllLines($batch,$lines,[Text.Encoding]::ASCII)
& cmd /c $batch
$result=$LASTEXITCODE
Get-Content (Join-Path $logs 'native-build.log') | Where-Object {{ $_ -notmatch '^[A-Z]:.*(cl|link)\\.exe' }} | Select-Object -Last 35
if ($result -ne 0) {{ throw "Native build failed: $result" }}
"""
        result = powershell(args.vm, code, timeout=900)
        print(result.stdout.strip(), flush=True)
        if result.stderr:
            print(result.stderr.strip(), flush=True)


if __name__ == "__main__":
    main()
