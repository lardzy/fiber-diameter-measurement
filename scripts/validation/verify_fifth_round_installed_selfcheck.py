"""Run the actual installed candidate's worker self-check as the current user."""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from verify_fifth_round_installer import (
    EXPECTED_ROOT, require, require_alignment_selfcheck, sha, write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    require(os.name == "nt" and str(root).casefold() == EXPECTED_ROOT.casefold(),
            "Use the isolated Windows fifth-round checkout")
    require(not ctypes.windll.shell32.IsUserAnAdmin(), "Run as the ordinary interactive user")
    identity = json.loads((root / "build/fifth-round/package/installer-identity.json").read_text(encoding="utf-8"))
    executable = root / "dist/windows/installed-candidate/FiberDiameterMeasurement.exe"
    expected = Path(identity["payload"]) / "FiberDiameterMeasurement.exe"
    require(identity.get("passed") is True and sha(executable) == sha(expected),
            "Installed executable must match the verified payload")
    manifest = json.loads((executable.parent / "release-manifest.json").read_text(encoding="utf-8"))
    require(manifest["build_id"] == identity["build_id"] and manifest["version"] == "0.4.10",
            "Installed build identity mismatch")
    out = root / "build/fifth-round/installed-selfcheck"
    require(not out.exists(), "Preserve prior installed self-check output")
    out.mkdir()
    with (out / "raw.stdout").open("wb") as stdout, (out / "stderr.log").open("wb") as stderr:
        subprocess.run([str(executable), "--self-check", "--json"],
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                       stdout=stdout, stderr=stderr, check=True, timeout=900)
    raw = (out / "raw.stdout").read_bytes()
    try:
        decoded, encoding = raw.decode("utf-8", errors="strict"), "utf-8"
    except UnicodeDecodeError:
        decoded, encoding = raw.decode("cp936", errors="strict"), "cp936"
    report = json.loads(decoded)
    require(report.get("ok") is True and not report.get("errors")
            and report.get("build_id") == identity["build_id"], "Installed self-check failed")
    alignment = require_alignment_selfcheck(report)
    write_json(out / "selfcheck.json", report)
    result = dict(passed=True, executable=str(executable), build_id=identity["build_id"],
                  executable_sha256=sha(executable), installed_application=True, elevated=False,
                  caller=os.environ.get("USERDOMAIN", "") + "\\" + os.environ.get("USERNAME", ""),
                  encoding=encoding, candidate_alignment_worker=alignment,
                  hardware_capture_tested=False, finished_at=datetime.now(timezone.utc).isoformat())
    write_json(out / "result.json", result)
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
